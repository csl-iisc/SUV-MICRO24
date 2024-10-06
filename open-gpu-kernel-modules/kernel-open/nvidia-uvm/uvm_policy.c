/*******************************************************************************
    Copyright (c) 2015-2022 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_ioctl.h"
#include "uvm_va_space.h"
#include "uvm_va_range.h"
#include "uvm_va_block.h"
#include "uvm_api.h"
#include "uvm_tracker.h"
#include "uvm_gpu.h"
#include "uvm_va_space_mm.h"

#include "uvm_gpu_access_counters.h"

bool uvm_is_valid_vma_range(struct mm_struct *mm, NvU64 start, NvU64 length)
{
    const NvU64 end = start + length;
    struct vm_area_struct *vma;

    UVM_ASSERT(mm);
    uvm_assert_mmap_lock_locked(mm);

    vma = find_vma_intersection(mm, start, end);

    while (vma && (vma->vm_start <= start)) {
        if (vma->vm_end >= end)
            return true;
        start = vma->vm_end;
        vma = find_vma_intersection(mm, start, end);
    }

    return false;
}

NV_STATUS uvm_api_range_type_check(uvm_va_space_t *va_space, struct mm_struct *mm, NvU64 base, NvU64 length)
{
    uvm_va_range_t *va_range, *va_range_last;
    const NvU64 last_address = base + length - 1;

    if (mm)
        uvm_assert_mmap_lock_locked(mm);

    uvm_assert_rwsem_locked(&va_space->lock);

    /* pr_alert("base = %xu\n", base); */
    /* pr_alert("length = %x\n", length); */
    if (uvm_api_range_invalid(base, length)) {
    /* pr_alert("0\n"); */
        return NV_ERR_INVALID_ADDRESS;
    }
    /* pr_alert("1\n"); */

    // Check if passed interval overlaps with any VA range.
    if (uvm_va_space_range_empty(va_space, base, last_address)) {
        if (g_uvm_global.ats.enabled &&
            uvm_va_space_pageable_mem_access_supported(va_space) &&
            mm &&
            uvm_is_valid_vma_range(mm, base, length))
            return NV_WARN_NOTHING_TO_DO;
        else if (uvm_hmm_is_enabled(va_space) &&
                 mm &&
                 uvm_is_valid_vma_range(mm, base, length))
            return NV_OK;
        else
            return NV_ERR_INVALID_ADDRESS;
    }
    /* pr_alert("2\n"); */

    va_range_last = NULL;

    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address)
        va_range_last = va_range;

    // Check if passed interval overlaps with an unmanaged VA range, or a
    // sub-interval not tracked by a VA range
    if (!va_range_last || va_range_last->node.end < last_address)
        return NV_ERR_INVALID_ADDRESS;

    /* pr_alert("3\n"); */
    // Passed interval is fully covered by managed VA ranges
    return NV_OK;
}

static NV_STATUS split_as_needed(uvm_va_space_t *va_space,
                                 NvU64 addr,
                                 uvm_va_policy_is_split_needed_t split_needed_cb,
                                 void *data)
{
    uvm_va_range_t *va_range;

    UVM_ASSERT(PAGE_ALIGNED(addr));

    // Look for managed allocations first, then look for HMM policies.
    va_range = uvm_va_range_find(va_space, addr);
    if (!va_range)
        return uvm_hmm_split_as_needed(va_space, addr, split_needed_cb, data);

    // If the policy range doesn't span addr, we're done.
    if (addr == va_range->node.start)
        return NV_OK;

    // Only managed ranges can be split.
    if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
        return NV_ERR_INVALID_ADDRESS;

    if (split_needed_cb(uvm_va_range_get_policy(va_range), data))
        return uvm_va_range_split(va_range, addr - 1, NULL);

    return NV_OK;
}

// Split policy ranges if split_needed_cb() returns true, where start_addr and
// end_addr (exclusive) define the process virtual address range.
// If splits are performed, start_addr and end_addr will be the starting
// and ending addresses of the newly-split policy range.
static NV_STATUS split_span_as_needed(uvm_va_space_t *va_space,
                                      NvU64 start_addr,
                                      NvU64 end_addr,
                                      uvm_va_policy_is_split_needed_t split_needed_cb,
                                      void *data)
{
    NV_STATUS status;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    status = split_as_needed(va_space, start_addr, split_needed_cb, data);
    if (status != NV_OK)
        return status;

    return split_as_needed(va_space, end_addr, split_needed_cb, data);
}

static bool ingore_notification_is_split_needed(uvm_va_policy_t *policy, void *data)
{
    bool ignore_ac_notification;

    UVM_ASSERT(data);

    ignore_ac_notification = *(bool*)data;
    return (ignore_ac_notification != policy->ignore_ac_notification);
}

static bool prioritized_location_is_split_needed(uvm_va_policy_t *policy, void *data)
{
    uvm_processor_id_t processor_id;

    UVM_ASSERT(data);

    processor_id = *(uvm_processor_id_t*)data;
    return !uvm_id_equal(processor_id, policy->prioritized_location);
}

static bool preferred_location_is_split_needed(uvm_va_policy_t *policy, void *data)
{
    uvm_processor_id_t processor_id;

    UVM_ASSERT(data);

    processor_id = *(uvm_processor_id_t*)data;
    return !uvm_id_equal(processor_id, policy->preferred_location);
}

static NV_STATUS preferred_location_unmap_remote_pages(uvm_va_block_t *va_block,
                                                       uvm_va_block_context_t *va_block_context)
{
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    uvm_va_policy_t *policy = va_block_context->policy;
    uvm_processor_id_t preferred_location = policy->preferred_location;
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block);
    const uvm_page_mask_t *mapped_mask;

    if (UVM_ID_IS_INVALID(preferred_location) || !uvm_processor_mask_test(&va_block->mapped, preferred_location))
        goto done;

    // Read duplication takes precedence over PreferredLocation. No mappings
    // need to be removed.
    if (uvm_va_policy_is_read_duplicate(policy, va_space))
        goto done;

    mapped_mask = uvm_va_block_map_mask_get(va_block, preferred_location);

    if (uvm_processor_mask_test(&va_block->resident, preferred_location)) {
        const uvm_page_mask_t *resident_mask = uvm_va_block_resident_mask_get(va_block, preferred_location);

        if (!uvm_page_mask_andnot(&va_block_context->caller_page_mask, mapped_mask, resident_mask))
            goto done;
    }
    else {
        uvm_page_mask_copy(&va_block_context->caller_page_mask, mapped_mask);
    }

    status = uvm_va_block_unmap(va_block,
                                va_block_context,
                                preferred_location,
                                uvm_va_block_region_from_block(va_block),
                                &va_block_context->caller_page_mask,
                                &local_tracker);

    tracker_status = uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
    if (status == NV_OK)
        status = tracker_status;

done:
    uvm_tracker_deinit(&local_tracker);

    return status;
}

NV_STATUS uvm_va_block_set_preferred_location_locked(uvm_va_block_t *va_block,
                                                     uvm_va_block_context_t *va_block_context)
{
    uvm_assert_mutex_locked(&va_block->lock);
    // TODO: Bug 1750144: remove this restriction when HMM handles setting
    // the preferred location semantics instead of just recording the policy.
    UVM_ASSERT(!uvm_va_block_is_hmm(va_block));
    UVM_ASSERT(va_block_context->policy == uvm_va_range_get_policy(va_block->va_range));

    uvm_va_block_mark_cpu_dirty(va_block);

    return preferred_location_unmap_remote_pages(va_block, va_block_context);
}

static NV_STATUS ignore_region_set(uvm_va_space_t *va_space,
                                   struct mm_struct *mm, NvU64 base,
                                   NvU64 length, bool ignore_ac_notification) {
  uvm_va_range_t *va_range, *va_range_last;
  const NvU64 last_address = base + length - 1;
  bool prioritized_location_is_faultable_gpu = false;
  NV_STATUS status;

  uvm_assert_rwsem_locked_write(&va_space->lock);

  status = split_span_as_needed(va_space, base, last_address + 1,
                                ingore_notification_is_split_needed,
                                &ignore_ac_notification);
  if (status != NV_OK)
    return status;

  va_range_last = NULL;
  uvm_for_each_managed_va_range_in_contig(va_range, va_space, base,
                                          last_address) {
    bool found_non_migratable_interval = false;

    va_range_last = va_range;

    // If we didn't split the ends, check that they match
    if (va_range->node.start < base || va_range->node.end > last_address)
      UVM_ASSERT( uvm_va_range_get_policy(va_range)->ignore_ac_notification == ignore_ac_notification);

    status = uvm_va_range_set_no_migrate_region(
        va_range, ignore_ac_notification); //  , mm, out_tracker);
    if (status != NV_OK)
      return status;

    /* // Return the first VA range that needs to be migrated so the caller */
    /* // function doesn't need to traverse the tree again */
    /* if (found_non_migratable_interval && (*first_va_range_to_migrate ==
     * NULL)) */
    /*     *first_va_range_to_migrate = va_range; */
  }

  if (va_range_last) {
    UVM_ASSERT(va_range_last->node.end >= last_address);
    return NV_OK;
  }

  /* return uvm_hmm_set_prioritized_location(va_space, prioritized_location,
   * base, last_address); */
  return NV_OK;
}

static NV_STATUS quick_migration_set(uvm_va_space_t *va_space,
                                        struct mm_struct *mm,
                                        NvU64 base,
                                        NvU64 length)
{
    uvm_va_range_t *va_range, *va_range_last;
    const NvU64 last_address = base + length - 1;
    NV_STATUS status;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        bool found_non_migratable_interval = false;

        va_range_last = va_range;

        status = uvm_va_range_set_quick_migrate(va_range, true);//  , mm, out_tracker);
        if (status != NV_OK)
            return status;
    }

    if (va_range_last) {
        UVM_ASSERT(va_range_last->node.end >= last_address);
        return NV_OK;
    }

    return NV_OK;
}

static NV_STATUS prioritized_location_set(uvm_va_space_t *va_space,
                                        struct mm_struct *mm,
                                        NvU64 base,
                                        NvU64 length,
                                        uvm_processor_id_t prioritized_location)
{
    uvm_va_range_t *va_range, *va_range_last;
    const NvU64 last_address = base + length - 1;
    bool prioritized_location_is_faultable_gpu = false;
    NV_STATUS status;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (UVM_ID_IS_VALID(prioritized_location)) {
        /* *first_va_range_to_migrate = NULL; */
        prioritized_location_is_faultable_gpu = UVM_ID_IS_GPU(prioritized_location) &&
            uvm_processor_mask_test(&va_space->faultable_processors,
                    prioritized_location);
    }

    status = split_span_as_needed(va_space,
            base,
            last_address + 1,
            prioritized_location_is_split_needed,
            &prioritized_location);
    if (status != NV_OK)
        return status;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        bool found_non_migratable_interval = false;

        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(uvm_id_equal(uvm_va_range_get_policy(va_range)->prioritized_location, prioritized_location));

        if (UVM_ID_IS_VALID(prioritized_location)) {
            const NvU64 start = max(base, va_range->node.start);
            const NvU64 end = min(last_address, va_range->node.end);

            found_non_migratable_interval = !uvm_range_group_all_migratable(va_space, start, end);

            if (found_non_migratable_interval && prioritized_location_is_faultable_gpu)
                return NV_ERR_INVALID_DEVICE;
        }

        status = uvm_va_range_set_prioritized_location(va_range, prioritized_location);//  , mm, out_tracker);
        if (status != NV_OK)
            return status;

        /* // Return the first VA range that needs to be migrated so the caller */
        /* // function doesn't need to traverse the tree again */
        /* if (found_non_migratable_interval && (*first_va_range_to_migrate == NULL)) */
        /*     *first_va_range_to_migrate = va_range; */

    }

    if (va_range_last) {
        UVM_ASSERT(va_range_last->node.end >= last_address);
        return NV_OK;
    }

    /* return uvm_hmm_set_prioritized_location(va_space, prioritized_location, base, last_address); */
    return NV_OK;
}

static NV_STATUS preferred_location_set(uvm_va_space_t *va_space,
                                        struct mm_struct *mm,
                                        NvU64 base,
                                        NvU64 length,
                                        uvm_processor_id_t preferred_location,
                                        uvm_va_range_t **first_va_range_to_migrate,
                                        uvm_tracker_t *out_tracker)
{
    uvm_va_range_t *va_range, *va_range_last;
    const NvU64 last_address = base + length - 1;
    bool preferred_location_is_faultable_gpu = false;
    NV_STATUS status;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (UVM_ID_IS_VALID(preferred_location)) {
        *first_va_range_to_migrate = NULL;
        preferred_location_is_faultable_gpu = UVM_ID_IS_GPU(preferred_location) &&
                                              uvm_processor_mask_test(&va_space->faultable_processors,
                                                                      preferred_location);
    }

    status = split_span_as_needed(va_space,
                                  base,
                                  last_address + 1,
                                  preferred_location_is_split_needed,
                                  &preferred_location);
    if (status != NV_OK)
        return status;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        bool found_non_migratable_interval = false;

        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(uvm_id_equal(uvm_va_range_get_policy(va_range)->preferred_location, preferred_location));

        if (UVM_ID_IS_VALID(preferred_location)) {
            const NvU64 start = max(base, va_range->node.start);
            const NvU64 end = min(last_address, va_range->node.end);

            found_non_migratable_interval = !uvm_range_group_all_migratable(va_space, start, end);

            if (found_non_migratable_interval && preferred_location_is_faultable_gpu)
                return NV_ERR_INVALID_DEVICE;
        }

        status = uvm_va_range_set_preferred_location(va_range, preferred_location, mm, out_tracker);
        if (status != NV_OK)
            return status;

        // Return the first VA range that needs to be migrated so the caller
        // function doesn't need to traverse the tree again
        if (found_non_migratable_interval && (*first_va_range_to_migrate == NULL))
            *first_va_range_to_migrate = va_range;
    }

    if (va_range_last) {
        UVM_ASSERT(va_range_last->node.end >= last_address);
        return NV_OK;
    }

    return uvm_hmm_set_preferred_location(va_space, preferred_location, base, last_address);
}

NV_STATUS uvm_api_set_preferred_location(const UVM_SET_PREFERRED_LOCATION_PARAMS *params, struct file *filp)
{
  pr_alert("setting prefferred location\n");

  /* return NV_OK; */
    NV_STATUS status;
    NV_STATUS tracker_status;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *first_va_range_to_migrate = NULL;
    struct mm_struct *mm;
    uvm_processor_id_t preferred_location_id;
    bool has_va_space_write_lock;
    const NvU64 start = params->requestedBase;
    const NvU64 length = params->length;
    const NvU64 end = start + length - 1;
    bool range_is_ats = false;

    UVM_ASSERT(va_space);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    has_va_space_write_lock = true;

    status = uvm_api_range_type_check(va_space, mm, start, length);
    if (status != NV_OK) {
        if (status != NV_WARN_NOTHING_TO_DO)
            goto done;

        status = NV_OK;
        range_is_ats = true;
    }

    // If the CPU is the preferred location, we don't have to find the associated uvm_gpu_t
    if (uvm_uuid_is_cpu(&params->preferredLocation)) {
        preferred_location_id = UVM_ID_CPU;
    }
    else {
        // Translate preferredLocation into a live GPU ID, and check that this
        // GPU can address the virtual address range
        uvm_gpu_t *gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->preferredLocation);

        if (!gpu)
            status = NV_ERR_INVALID_DEVICE;
        else if (!uvm_gpu_can_address(gpu, start, length))
            status = NV_ERR_OUT_OF_RANGE;

        if (status != NV_OK)
            goto done;

        preferred_location_id = gpu->id;
    }

    UVM_ASSERT(status == NV_OK);

    // TODO: Bug 2098544: On ATS systems, honor the preferred location policy
    // for system memory ranges instead of ignoring it.
    if (range_is_ats)
        goto done;

    status = preferred_location_set(va_space, mm, start, length, preferred_location_id, &first_va_range_to_migrate, &local_tracker);
    if (status != NV_OK)
        goto done;

    // No VA range to migrate, early exit
    if (!first_va_range_to_migrate){
    pr_alert("NOT needing to migrate\n");
        goto done;
    }

    uvm_va_space_downgrade_write(va_space);
    has_va_space_write_lock = false;

    pr_alert("needing to migrate\n");
    // No need to check for holes in the VA ranges span here, this was checked by preferred_location_set
    for (va_range = first_va_range_to_migrate; va_range; va_range = uvm_va_space_iter_next(va_range, end)) {
        uvm_range_group_range_iter_t iter;
        NvU64 cur_start = max(start, va_range->node.start);
        NvU64 cur_end = min(end, va_range->node.end);

        uvm_range_group_for_each_migratability_in(&iter, va_space, cur_start, cur_end) {
            if (!iter.migratable) {
                status = uvm_range_group_va_range_migrate(va_range, iter.start, iter.end, &local_tracker);
                if (status != NV_OK)
                    goto done;
            }
        }
    }

done:
    tracker_status = uvm_tracker_wait_deinit(&local_tracker);

    if (has_va_space_write_lock)
        uvm_va_space_up_write(va_space);
    else
        uvm_va_space_up_read(va_space);

    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_api_set_quick_migration(const UVM_SET_QUICK_MIGRATE_REGION_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_va_range_t *va_range = NULL;
    struct mm_struct *mm;
    bool has_va_space_write_lock;
    const NvU64 start = params->requestedBase;
    const NvU64 length = params->length;
    const NvU64 end = start + length - 1;
    bool range_is_ats = false;
    UVM_ASSERT(va_space);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    has_va_space_write_lock = true;

    status = uvm_api_range_type_check(va_space, mm, start, length);
    if (status != NV_OK) {
    pr_alert("sqm not okay\n");
        if (status != NV_WARN_NOTHING_TO_DO)
            goto done;

        status = NV_OK;
        range_is_ats = true;
    }

    UVM_ASSERT(status == NV_OK);

    if (range_is_ats)
        goto done;

    status = quick_migration_set(va_space, mm, start, length);
    if (status != NV_OK)
        goto done;

    uvm_va_space_downgrade_write(va_space);
    has_va_space_write_lock = false;

done:

    if (has_va_space_write_lock)
        uvm_va_space_up_write(va_space);
    else
        uvm_va_space_up_read(va_space);

    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return status;
}

NV_STATUS uvm_api_set_prioritized_location(const UVM_SET_PRIORITIZED_LOCATION_PARAMS *params, struct file *filp)
{
  /* pr_alert("PRIORTY API CALLED\n"); */
  /* pr_alert("temporarily using to print num AC \n"); */
/* print_ac_previous(); */
/* return NV_OK; */

    NV_STATUS status;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_va_range_t *va_range = NULL;
    struct mm_struct *mm;
    uvm_processor_id_t prioritized_location_id;
    bool has_va_space_write_lock;
    const NvU64 start = params->requestedBase;
    const NvU64 length = params->length;
    const NvU64 end = start + length - 1;
    bool range_is_ats = false;
    UVM_ASSERT(va_space);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    has_va_space_write_lock = true;

    status = uvm_api_range_type_check(va_space, mm, start, length);
    if (status != NV_OK) {
        if (status != NV_WARN_NOTHING_TO_DO)
            goto done;

        status = NV_OK;
        range_is_ats = true;
    }
    if (uvm_uuid_is_cpu(&params->prioritizedLocation)) {
        prioritized_location_id = UVM_ID_CPU;
    }
    else {
        uvm_gpu_t *gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->prioritizedLocation);

        if (!gpu)
            status = NV_ERR_INVALID_DEVICE;
        else if (!uvm_gpu_can_address(gpu, start, length))
            status = NV_ERR_OUT_OF_RANGE;

        if (status != NV_OK)
            goto done;

        prioritized_location_id = gpu->id;
    }

    UVM_ASSERT(status == NV_OK);

    if (range_is_ats)
        goto done;

    status = prioritized_location_set(va_space, mm, start, length, prioritized_location_id);
    if (status != NV_OK)
        goto done;

    uvm_va_space_downgrade_write(va_space);
    has_va_space_write_lock = false;

done:

    if (has_va_space_write_lock)
        uvm_va_space_up_write(va_space);
    else
        uvm_va_space_up_read(va_space);

    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return status;

    /* return NV_OK; */
}

NV_STATUS
uvm_api_set_no_migrate_region(const UVM_SET_NO_MIGRATE_REGION_PARAMS *params,
                              struct file *filp) {
  NV_STATUS status;
  pr_alert("NO MIGRATE API CALLED\n");
  uvm_va_space_t *va_space = uvm_va_space_get(filp);
  uvm_va_range_t *va_range = NULL;
  struct mm_struct *mm;
  //   uvm_processor_id_t prioritized_location_id;
  bool ignore_ac_notification;
  bool has_va_space_write_lock;
  const NvU64 start = params->requestedBase;
  const NvU64 length = params->length;
  const NvU64 end = start + length - 1;
  bool range_is_ats = false;
  UVM_ASSERT(va_space);

  mm = uvm_va_space_mm_or_current_retain_lock(va_space);
  uvm_va_space_down_write(va_space);
  has_va_space_write_lock = true;

  status = uvm_api_range_type_check(va_space, mm, start, length);
  if (status != NV_OK) {
    if (status != NV_WARN_NOTHING_TO_DO)
      goto done;

    status = NV_OK;
    range_is_ats = true;
  }

  ignore_ac_notification = &params->ignore_ac_notification;

  if (range_is_ats)
    goto done;

  status = ignore_region_set(va_space, mm, start, length,
                                    ignore_ac_notification);
  if (status != NV_OK)
    goto done;

  uvm_va_space_downgrade_write(va_space);
  has_va_space_write_lock = false;

done:

  if (has_va_space_write_lock)
    uvm_va_space_up_write(va_space);
  else
    uvm_va_space_up_read(va_space);

  uvm_va_space_mm_or_current_release_unlock(va_space, mm);

  return status;
}

NV_STATUS uvm_api_unset_preferred_location(const UVM_UNSET_PREFERRED_LOCATION_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    NV_STATUS tracker_status;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

    UVM_ASSERT(va_space);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    status = uvm_api_range_type_check(va_space, mm, params->requestedBase, params->length);

    if (status == NV_OK)
        status = preferred_location_set(va_space, mm, params->requestedBase, params->length, UVM_ID_INVALID, NULL, &local_tracker);
    else if (status == NV_WARN_NOTHING_TO_DO)
        status = NV_OK;

    tracker_status = uvm_tracker_wait_deinit(&local_tracker);

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    return status == NV_OK ? tracker_status : status;
}

static NV_STATUS va_block_set_accessed_by_locked(uvm_va_block_t *va_block,
                                                 uvm_va_block_context_t *va_block_context,
                                                 uvm_processor_id_t processor_id,
                                                 uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    NV_STATUS tracker_status;

    uvm_assert_mutex_locked(&va_block->lock);

    status = uvm_va_block_add_mappings(va_block,
                                       va_block_context,
                                       processor_id,
                                       uvm_va_block_region_from_block(va_block),
                                       NULL,
                                       UvmEventMapRemoteCausePolicy);

    tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &va_block->tracker);

    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_va_block_set_accessed_by(uvm_va_block_t *va_block,
                                       uvm_va_block_context_t *va_block_context,
                                       uvm_processor_id_t processor_id)
{
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block);
    NV_STATUS status;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

    UVM_ASSERT(!uvm_va_block_is_hmm(va_block));
    UVM_ASSERT(va_block_context->policy == uvm_va_range_get_policy(va_block->va_range));

    // Read duplication takes precedence over SetAccessedBy. Do not add mappings
    // if read duplication is enabled.
    if (uvm_va_policy_is_read_duplicate(va_block_context->policy, va_space))
        return NV_OK;

    status = UVM_VA_BLOCK_LOCK_RETRY(va_block, NULL,
                                     va_block_set_accessed_by_locked(va_block,
                                                                     va_block_context,
                                                                     processor_id,
                                                                     &local_tracker));

    // TODO: Bug 1767224: Combine all accessed_by operations into single tracker
    if (status == NV_OK)
        status = uvm_tracker_wait(&local_tracker);

    uvm_tracker_deinit(&local_tracker);
    return status;
}

typedef struct
{
    uvm_processor_id_t processor_id;
    bool set_bit;
} accessed_by_split_params_t;

static bool accessed_by_is_split_needed(uvm_va_policy_t *policy, void *data)
{
    accessed_by_split_params_t *params = (accessed_by_split_params_t*)data;

    UVM_ASSERT(params);

    return (uvm_processor_mask_test(&policy->accessed_by, params->processor_id) != params->set_bit);
}

static NV_STATUS accessed_by_set(uvm_va_space_t *va_space,
                                 NvU64 base,
                                 NvU64 length,
                                 const NvProcessorUuid *processor_uuid,
                                 bool set_bit)
{
    uvm_processor_id_t processor_id = UVM_ID_INVALID;
    uvm_va_range_t *va_range, *va_range_last;
    struct mm_struct *mm;
    const NvU64 last_address = base + length - 1;
    bool range_is_sysmem = false;
    accessed_by_split_params_t split_params;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS status;
    NV_STATUS tracker_status;

    UVM_ASSERT(va_space);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    status = uvm_api_range_type_check(va_space, mm, base, length);
    if (status != NV_OK) {
        if (status != NV_WARN_NOTHING_TO_DO)
            goto done;
        status = NV_OK;
        range_is_sysmem = true;
    }

    if (uvm_uuid_is_cpu(processor_uuid)) {
        processor_id = UVM_ID_CPU;
    }
    else {
        // Translate processor_uuid into a live GPU ID, and check that this GPU
        // can address the virtual address range
        uvm_gpu_t *gpu = uvm_va_space_get_gpu_by_uuid(va_space, processor_uuid);
        if (!gpu)
            status = NV_ERR_INVALID_DEVICE;
        else if (!uvm_gpu_can_address(gpu, base, length))
            status = NV_ERR_OUT_OF_RANGE;

        if (status != NV_OK)
            goto done;

        processor_id = gpu->id;
    }

    if (range_is_sysmem)
        goto done;

    split_params.processor_id = processor_id;
    split_params.set_bit = set_bit;
    status = split_span_as_needed(va_space,
                                  base,
                                  last_address + 1,
                                  accessed_by_is_split_needed,
                                  &split_params);
    if (status != NV_OK)
        goto done;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(uvm_processor_mask_test(&uvm_va_range_get_policy(va_range)->accessed_by,
                                               processor_id) == set_bit);

        if (set_bit) {
            status = uvm_va_range_set_accessed_by(va_range, processor_id, mm, &local_tracker);
            if (status != NV_OK)
                goto done;
        }
        else {
            uvm_va_range_unset_accessed_by(va_range, processor_id, &local_tracker);
        }
    }

    if (va_range_last) {
        UVM_ASSERT(va_range_last->node.end >= last_address);
        goto done;
    }

    status = uvm_hmm_set_accessed_by(va_space,
                                     processor_id,
                                     set_bit,
                                     base,
                                     last_address);

done:
    tracker_status = uvm_tracker_wait_deinit(&local_tracker);

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_api_set_accessed_by(const UVM_SET_ACCESSED_BY_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return accessed_by_set(va_space, params->requestedBase, params->length, &params->accessedByUuid, true);
}

NV_STATUS uvm_api_unset_accessed_by(const UVM_UNSET_ACCESSED_BY_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return accessed_by_set(va_space, params->requestedBase, params->length, &params->accessedByUuid, false);
}

static NV_STATUS va_block_set_read_duplication_locked(uvm_va_block_t *va_block,
                                                      uvm_va_block_retry_t *va_block_retry,
                                                      uvm_va_block_context_t *va_block_context)
{
    uvm_processor_id_t src_id;

    uvm_assert_mutex_locked(&va_block->lock);

    for_each_id_in_mask(src_id, &va_block->resident) {
        NV_STATUS status;
        uvm_page_mask_t *resident_mask = uvm_va_block_resident_mask_get(va_block, src_id);

        // Calling uvm_va_block_make_resident_read_duplicate will break all
        // SetAccessedBy and remote mappings
        status = uvm_va_block_make_resident_read_duplicate(va_block,
                                                           va_block_retry,
                                                           va_block_context,
                                                           src_id,
                                                           uvm_va_block_region_from_block(va_block),
                                                           resident_mask,
                                                           NULL,
                                                           UVM_MAKE_RESIDENT_CAUSE_API_HINT);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

NV_STATUS uvm_va_block_set_read_duplication(uvm_va_block_t *va_block,
                                            uvm_va_block_context_t *va_block_context)
{
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;

    // TODO: Bug 3660922: need to implement HMM read duplication support.
    UVM_ASSERT(!uvm_va_block_is_hmm(va_block));
    UVM_ASSERT(va_block_context->policy == uvm_va_range_get_policy(va_block->va_range));

    status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                                     va_block_set_read_duplication_locked(va_block,
                                                                          &va_block_retry,
                                                                          va_block_context));

    return status;
}

static NV_STATUS va_block_unset_read_duplication_locked(uvm_va_block_t *va_block,
                                                        uvm_va_block_retry_t *va_block_retry,
                                                        uvm_va_block_context_t *va_block_context,
                                                        uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    uvm_processor_id_t processor_id;
    uvm_va_block_region_t block_region = uvm_va_block_region_from_block(va_block);
    uvm_page_mask_t *break_read_duplication_pages = &va_block_context->caller_page_mask;
    uvm_va_policy_t *policy = va_block_context->policy;
    uvm_processor_id_t preferred_location = policy->preferred_location;
    uvm_processor_mask_t accessed_by = policy->accessed_by;

    uvm_assert_mutex_locked(&va_block->lock);

    // 1- Iterate over all processors with resident copies to avoid migrations
    // and invalidate the rest of copies

    // If preferred_location is set and has resident copies, give it preference
    if (UVM_ID_IS_VALID(preferred_location) &&
        uvm_processor_mask_test(&va_block->resident, preferred_location)) {
        uvm_page_mask_t *resident_mask = uvm_va_block_resident_mask_get(va_block, preferred_location);
        bool is_mask_empty = !uvm_page_mask_and(break_read_duplication_pages,
                                                &va_block->read_duplicated_pages,
                                                resident_mask);

        if (!is_mask_empty) {
            // make_resident breaks read duplication
            status = uvm_va_block_make_resident(va_block,
                                                va_block_retry,
                                                va_block_context,
                                                preferred_location,
                                                block_region,
                                                break_read_duplication_pages,
                                                NULL,
                                                UVM_MAKE_RESIDENT_CAUSE_API_HINT);
            if (status != NV_OK)
                return status;
        }
    }

    // Then iterate over the rest of processors
    for_each_id_in_mask(processor_id, &va_block->resident) {
        uvm_page_mask_t *resident_mask;
        bool is_mask_empty;

        if (uvm_id_equal(processor_id, preferred_location))
            continue;

        resident_mask = uvm_va_block_resident_mask_get(va_block, processor_id);
        is_mask_empty = !uvm_page_mask_and(break_read_duplication_pages,
                                           &va_block->read_duplicated_pages,
                                           resident_mask);
        if (is_mask_empty)
            continue;

        // make_resident breaks read duplication
        status = uvm_va_block_make_resident(va_block,
                                            va_block_retry,
                                            va_block_context,
                                            processor_id,
                                            block_region,
                                            break_read_duplication_pages,
                                            NULL,
                                            UVM_MAKE_RESIDENT_CAUSE_API_HINT);
        if (status != NV_OK)
            return status;
    }

    // 2- Re-establish SetAccessedBy mappings
    for_each_id_in_mask(processor_id, &accessed_by) {
        status = va_block_set_accessed_by_locked(va_block,
                                                 va_block_context,
                                                 processor_id,
                                                 out_tracker);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

NV_STATUS uvm_va_block_unset_read_duplication(uvm_va_block_t *va_block,
                                              uvm_va_block_context_t *va_block_context)
{
    uvm_va_block_retry_t va_block_retry;
    NV_STATUS status = NV_OK;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

    UVM_ASSERT(!uvm_va_block_is_hmm(va_block));
    UVM_ASSERT(va_block_context->policy == uvm_va_range_get_policy(va_block->va_range));

    // Restore all SetAccessedBy mappings
    status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                                     va_block_unset_read_duplication_locked(va_block,
                                                                            &va_block_retry,
                                                                            va_block_context,
                                                                            &local_tracker));
    if (status == NV_OK)
        status = uvm_tracker_wait(&local_tracker);

    uvm_tracker_deinit(&local_tracker);

    return status;
}

static bool read_duplication_is_split_needed(uvm_va_policy_t *policy, void *data)
{
    uvm_read_duplication_policy_t new_policy;

    UVM_ASSERT(data);

    new_policy = *(uvm_read_duplication_policy_t *)data;
    return policy->read_duplication != new_policy;
}

static NV_STATUS read_duplication_set(uvm_va_space_t *va_space, NvU64 base, NvU64 length, bool enable)
{
    uvm_va_range_t *va_range, *va_range_last;
    struct mm_struct *mm;
    const NvU64 last_address = base + length - 1;
    NV_STATUS status;
    uvm_read_duplication_policy_t new_policy;

    UVM_ASSERT(va_space);

    // We need mmap_lock as we may create CPU mappings
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    status = uvm_api_range_type_check(va_space, mm, base, length);
    if (status != NV_OK) {
        if (status == NV_WARN_NOTHING_TO_DO)
            status = NV_OK;

        goto done;
    }

    // Note that we never set the policy back to UNSET
    new_policy = enable ? UVM_READ_DUPLICATION_ENABLED : UVM_READ_DUPLICATION_DISABLED;

    status = split_span_as_needed(va_space,
                                  base,
                                  last_address + 1,
                                  read_duplication_is_split_needed,
                                  &new_policy);
    if (status != NV_OK)
        goto done;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(uvm_va_range_get_policy(va_range)->read_duplication == new_policy);

        // If the va_space cannot currently read duplicate, only change the user
        // state. All memory should already have read duplication unset.
        if (uvm_va_space_can_read_duplicate(va_space, NULL)) {

            // Handle SetAccessedBy mappings
            if (new_policy == UVM_READ_DUPLICATION_ENABLED) {
                status = uvm_va_range_set_read_duplication(va_range, mm);
                if (status != NV_OK)
                    goto done;
            }
            else {
                // If unsetting read duplication fails, the return status is
                // not propagated back to the caller
                (void)uvm_va_range_unset_read_duplication(va_range, mm);
            }
        }

        uvm_va_range_get_policy(va_range)->read_duplication = new_policy;
    }

    if (va_range_last) {
        UVM_ASSERT(va_range_last->node.end >= last_address);
        goto done;
    }

    status = uvm_hmm_set_read_duplication(va_space,
                                          new_policy,
                                          base,
                                          last_address);

done:
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    return status;
}

NV_STATUS uvm_api_enable_read_duplication(const UVM_ENABLE_READ_DUPLICATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return read_duplication_set(va_space, params->requestedBase, params->length, true);
}

NV_STATUS uvm_api_disable_read_duplication(const UVM_DISABLE_READ_DUPLICATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return read_duplication_set(va_space, params->requestedBase, params->length, false);
}

static NV_STATUS system_wide_atomics_set(uvm_va_space_t *va_space, const NvProcessorUuid *gpu_uuid, bool enable)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    bool already_enabled;

    uvm_va_space_down_write(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    if (gpu->parent->scoped_atomics_supported) {
        status = NV_ERR_NOT_SUPPORTED;
        goto done;
    }

    if (!uvm_processor_mask_test(&va_space->faultable_processors, gpu->id)) {
        status = NV_ERR_NOT_SUPPORTED;
        goto done;
    }

    already_enabled = uvm_processor_mask_test(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    if (enable && !already_enabled) {
        uvm_va_range_t *va_range;
        uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
        uvm_va_block_context_t *va_block_context = uvm_va_space_block_context(va_space, NULL);
        NV_STATUS tracker_status;

        // Revoke atomic mappings from the calling GPU
        uvm_for_each_va_range(va_range, va_space) {
            uvm_va_block_t *va_block;

            if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
                continue;

            va_block_context->policy = uvm_va_range_get_policy(va_range);
            for_each_va_block_in_va_range(va_range, va_block) {
                uvm_page_mask_t *non_resident_pages = &va_block_context->caller_page_mask;

                uvm_mutex_lock(&va_block->lock);

                if (!uvm_processor_mask_test(&va_block->mapped, gpu->id)) {
                    uvm_mutex_unlock(&va_block->lock);
                    continue;
                }

                uvm_page_mask_complement(non_resident_pages, &va_block->gpus[uvm_id_gpu_index(gpu->id)]->resident);

                status = uvm_va_block_revoke_prot(va_block,
                                                  va_block_context,
                                                  gpu->id,
                                                  uvm_va_block_region_from_block(va_block),
                                                  non_resident_pages,
                                                  UVM_PROT_READ_WRITE_ATOMIC,
                                                  &va_block->tracker);

                tracker_status = uvm_tracker_add_tracker_safe(&local_tracker, &va_block->tracker);

                uvm_mutex_unlock(&va_block->lock);

                if (status == NV_OK)
                    status = tracker_status;

                if (status != NV_OK) {
                    uvm_tracker_deinit(&local_tracker);
                    goto done;
                }
            }
        }
        status = uvm_tracker_wait_deinit(&local_tracker);

        uvm_processor_mask_set(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    }
    else if (!enable && already_enabled) {
        // TODO: Bug 1767229: Promote write mappings to atomic
        uvm_processor_mask_clear(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    }

done:
    uvm_va_space_up_write(va_space);
    return status;
}

NV_STATUS uvm_api_enable_system_wide_atomics(UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return system_wide_atomics_set(va_space, &params->gpu_uuid, true);
}

NV_STATUS uvm_api_disable_system_wide_atomics(UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return system_wide_atomics_set(va_space, &params->gpu_uuid, false);
}
