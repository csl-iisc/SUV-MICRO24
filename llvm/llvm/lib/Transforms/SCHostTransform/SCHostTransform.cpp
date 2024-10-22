//===- SCHostTransform.cpp - Extracting access information from CUDA
// kernels
//---------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "SCHostTransform"

// The following line is edited by scripts to set the GPU size.
unsigned long long GPU_SIZE = (1ULL) * 1024ULL * 1024ULL * 2048ULL;
double MIN_ALLOC_PERC = 6;

namespace {

enum AllocationAccessPatternType {
  AAPT_HIGH_PHI,
  AAPT_HIGH_X,
  AAPT_HIGH_Y,
  AAPT_NONE,
};

enum ExprTreeOp {
  ETO_PC,
  ETO_ADD,
  ETO_SUB,
  ETO_AND,
  ETO_OR,
  ETO_MUL,
  ETO_DIV,
  ETO_UDIV,
  ETO_SDIV,
  ETO_SREM,
  ETO_FMUL,
  ETO_FDIV,
  ETO_SHL,
  ETO_LSHR,
  ETO_DOUBLE,
  ETO_PHI,
  ETO_ICMP,
  ETO_FCMP,
  ETO_MEMOP,
  ETO_CONST,
  ETO_PHI_TERM,
  ETO_BDIMX,
  ETO_BDIMY,
  ETO_BIDX,
  ETO_BIDY,
  ETO_TIDX,
  ETO_TIDY,
  ETO_ARG,
  ETO_GEP,
  ETO_ZEXT,
  ETO_SEXT,
  ETO_FREEZE,
  ETO_TRUNC,
  ETO_FPTOSI,
  ETO_UITOFP,
  ETO_SITOFP,
  ETO_SELECT,
  ETO_ATOMICRMW,
  ETO_UNDEF,
  ETO_INCOMP,
  ETO_UNKNOWN,
  ETO_CALL,
  ETO_INTERM,
  ETO_LOAD,
  ETO_NONE,
};

struct ExprTreeNode {
  ExprTreeOp op;
  /* Value* val; */
  unsigned arg;
  unsigned long long value;
  std::string original_str;
  struct ExprTreeNode *parent;
  struct ExprTreeNode *children[2];
  bool isProb = false;
};

struct ExprTreeNodeAdvanced {
  ExprTreeOp op;
  unsigned arg;
  unsigned long long value;
  std::string original_str;
  struct ExprTreeNodeAdvanced *parent;
  std::vector<ExprTreeNodeAdvanced*> children;
  bool isProb = false;
};

enum AdvisoryType {
  ADVISORY_SET_PREFERRED_LOCATION,
  ADVISORY_SET_ACCESSED_BY,
  ADVISORY_SET_PRIORITIZED_LOCATION,
  ADVISORY_SET_PREFETCH,
  ADVISORY_SET_PIN_HOST,
  ADVISORY_SET_PIN_DEVICE,
  ADVISORY_SET_DEMAND_MIGRATE,
  ADVISORY_MAX,
};

enum BlockSizeType {
  AXIS_TYPE_BDIMX,
  AXIS_TYPE_BDIMY,
  AXIS_TYPE_BDIMZ,
  AXIS_TYPE_NONE,
};

enum GridSizeType {
  AXIS_TYPE_GDIMX,
  AXIS_TYPE_GDIMY,
  AXIS_TYPE_GDIMZ,
  AXIS_TYPE_GNONE,
};

std::map<std::string, BlockSizeType> StringToBlockSizeType = {
    {std::string("SREG_BDIMX"), AXIS_TYPE_BDIMX},
    {std::string("SREG_BDIMY"), AXIS_TYPE_BDIMY},
    {std::string("SREG_BDIMZ"), AXIS_TYPE_BDIMZ},
};

enum IndexAxisType {
  INDEX_AXIS_LOOPVAR,
  INDEX_AXIS_BIDX,
  INDEX_AXIS_BIDY,
  INDEX_AXIS_BIDZ,
  INDEX_AXIS_MAX,
};

std::map<std::string, IndexAxisType> StringToIndexAxisType = {
    {std::string("LOOPVAR"), INDEX_AXIS_LOOPVAR},
    {std::string("BIDX"), INDEX_AXIS_BIDX},
    {std::string("BIDY"), INDEX_AXIS_BIDY},
    {std::string("BIDZ"), INDEX_AXIS_BIDZ},
};

std::map<IndexAxisType, std::string> IndexAxisTypeToString = {
    {INDEX_AXIS_LOOPVAR, std::string("LOOPVAR")},
    {INDEX_AXIS_BIDX, std::string("BIDX")},
    {INDEX_AXIS_BIDY, std::string("BIDY")},
    {INDEX_AXIS_BIDZ, std::string("BIDZ")},
};

struct IndexAxisMultiplier {
  IndexAxisType IndexAxis;
  int Multiplier;
};

struct SubAllocationStruct {
  AdvisoryType Advisory;
  unsigned long long StartIndex, Size;
  unsigned long long PrefetchIters, PrefetchSize;
};

// represents the properties of a memalloc
struct AllocationStruct {
  Value *AllocationInst;
  unsigned long long AccessCount;
  unsigned long long Size;
  float Density;
  unsigned long long wss;
  unsigned pd_phi, pd_bidx, pd_bidy;
  // std::map<IndexAxisType, unsigned> IndexStride;
  std::vector<unsigned> IndexAxisConstants; // (INDEX_AXIS_MAX);
  AdvisoryType Advisory;
  unsigned long long AdvisorySize;
  AllocationAccessPatternType AAPType;
  std::vector<SubAllocationStruct *> SubAllocations;
  bool isPC;
  // Constructor below
  AllocationStruct() : Advisory(ADVISORY_MAX){};
};

// reverse sorting
bool allocationSorter(AllocationStruct const &Lhs,
                      AllocationStruct const &Rhs) {
  return Lhs.Density > Rhs.Density;
}

/* std::vector<struct AllocationStruct> AllocationStructs; */
    bool multiKernel = false;

std::set<Value *> StructAllocas;
std::map<AllocaInst *, std::map<unsigned, Value *>>
    StructAllocasToIndexToValuesMap;

std::set<Function *> ListOfLocallyDefinedFunctions;
std::map<Function *, std::vector<Value *>> FunctionToFormalArgumentMap;
std::map<CallBase *, std::vector<Value *>> FunctionCallToActualArumentsMap;
std::map<Value *, std::vector<Value *>> FormalArgumentToActualArgumentMap;
/* std::map<Value *, Value *> ActualArgumentToFormalArgumentMap; */
std::map<Value *, std::map<Value *, Value *>>
    FunctionCallToFormalArgumentToActualArgumentMap;
std::map<Value *, std::map<Value *, Value *>>
    FunctionCallToActualArgumentToFormalArgumentMap;

std::set<Value *> OriginalPointers;
std::map<Value *, Value *> PointerOpToOriginalPointers;
std::map<Value *, Value *> PointerOpToOriginalStructPointer;
std::map<Value *, unsigned> PointerOpToOriginalStructPointersIndex;

std::map<Value *, unsigned> PointerOpToOriginalConstant;

std::set<CallBase *> VisitedCallInstForPointerPropogation;

std::set<Instruction *> MemcpyOpForStructs;
std::map<Value *, Instruction *> MemcpyOpForStructsSrcToInstMap;
std::map<Value *, Instruction *> MemcpyOpForStructsDstToInstMap;

// for each kernel invocation, for each argument, store the access count
std::map<std::string, std::vector<std::pair<unsigned, unsigned>>>
    KernelParamUsageInKernel;
// for each kernel invocation, for each argument, for differnt axes, store the
// multipliers
std::map<std::string,
         std::map<unsigned, std::map<IndexAxisType, std::vector<std::string>>>>
    KernelParamReuseInKernel;
std::map<Value *, unsigned long int> MallocSizeMap;
std::map<Value *, unsigned long int> MallocPointerToSizeMap;
std::map<Value *, std::map<unsigned, unsigned long long>>
    MallocPointerStructToIndexToSizeMap;
std::set<Value *> MallocPointers;

std::map<Value *, std::vector<Value *>> KernelArgToStoreMap;
std::map<Instruction *, Value *> KernelInvocationToStructMap;
std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToActualArgMap;
std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToAllocationMap;
std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToLastStoreMap;
std::map<Instruction *, std::map<Value *, Value *>>
    KernelInvocationToKernArgToAllocationMap;
std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToConstantMap;
std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToLIVMap;
std::map<Instruction *, std::map<Value *, unsigned>>
    KernelInvocationToLIVToArgNumMap;
std::map<Instruction *, std::map<BlockSizeType, unsigned>>
    KernelInvocationToBlockSizeMap;
std::map<Instruction *, std::map<GridSizeType, unsigned>>
    KernelInvocationToGridSizeMap; // when grid size is constant
std::map<Instruction *, std::map<GridSizeType, Value *>>
    KernelInvocationToGridSizeValueMap; // when grid size is variable

    std::map<Value*, Value*> AllocationToFirstMap;

std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToAllocationArgNumberToKernelArgMap;

std::map<Instruction *, Value *> KernelInvocationToGridDimXYValueMap;
std::map<Instruction *, Value *> KernelInvocationToGridDimZValueMap;
// std::map<Instruction*, Value*> KernelInvocationToGridDimXYValueMap;
// std::map<Instruction*, Value*> KernelInvocationToGridDimZValueMap;

// InvocationId is used to identify each kernel invocation
// starts at 1, unique for each kernel invocation
static unsigned KernelInvocationID = 1;

std::map<Instruction*, unsigned> KernelInvocationToInvocationIDMap;
std::map<Instruction *, unsigned long> KernelInvocationToIterMap;
std::map<Instruction *, unsigned long> KernelInvocationToStepsMap;

std::map<Instruction *, std::map<unsigned, unsigned long long>>
    KernelInvocationToAccessIDToAccessDensity;
std::map<Instruction *, std::map<unsigned, unsigned>>
    KernelInvocationToAccessIDToPartDiff_phi;
std::map<Instruction *, std::map<unsigned, unsigned>>
    KernelInvocationToAccessIDToPartDiff_bidx;
std::map<Instruction *, std::map<unsigned, unsigned>>
    KernelInvocationToAccessIDToPartDiff_bidy;
std::map<Instruction *, std::map<unsigned, unsigned>>
    KernelInvocationToAccessIDToPartDiff_looparg;
std::map<Instruction *, std::map<unsigned, unsigned>>
    KernelInvocationToAccessIDToWSS;

std::map<Instruction *, Instruction *> KernelInvocationToEnclosingLIVMap;
std::map<Instruction *, Instruction *> KernelInvocationToEnclosingLoopPredMap;
std::map<Instruction *, Function *> KernelInvocationToEnclosingFunction;

// map from loop id to loop iterations
std::map<std::string, std::map<unsigned, std::vector<std::string>>> LoopIDToLoopBoundsMap;
std::map<std::string, std::map<unsigned, unsigned>> LoopIDToLoopItersMap;
std::map<unsigned, unsigned> LoopIDToParentLoopIDMap; // loop IDs are unique across kernels
std::map<unsigned, unsigned> PhiNodeToLoopIDMap; // phi node to nearest enclosing loop, if any


std::map<unsigned, std::vector<std::string>> IfIDToCondMap;

std::map<std::string, std::map<unsigned, ExprTreeNode *>>
    LoopIDToBoundsExprMapIn;
std::map<std::string, std::map<unsigned, ExprTreeNode *>>
    LoopIDToBoundsExprMapFin;
std::map<std::string, std::map<unsigned, ExprTreeNode *>>
    LoopIDToBoundsExprMapStep;
std::map<std::string, std::map<unsigned, unsigned>> LoopIDToBoundsMapIn;
std::map<std::string, std::map<unsigned, unsigned>> LoopIDToBoundsMapFin;
std::map<std::string, std::map<unsigned, unsigned>> LoopIDToBoundsMapStep;

// map from access id to expression tree
// TODO: these names need to change to include kernel name in them
std::map<std::string, std::map<unsigned, unsigned>>
    KernelNameToAccessIDToAllocationArgMap;
std::map<std::string, std::map<unsigned, unsigned>>
    KernelNameToAccessIDToEnclosingLoopMap;
std::map<std::string, std::map<unsigned, ExprTreeNode *>>
    KernelNameToAccessIDToExpressionTreeMap;
std::map<std::string, std::map<unsigned, ExprTreeNodeAdvanced *>>
    KernelNameToAccessIDToAdvancedExpressionTreeMap;
std::map<std::string, std::map<unsigned, unsigned>>
    KernelNameToAccessIDToIfCondMap;
std::map<std::string, std::map<unsigned, unsigned>>
    KernelNameToAccessIDToIfTypeMap;

std::set<ExprTreeOp> terminals;
std::set<ExprTreeOp> operations;

// terminal values are like kernel-arguments,
std::set<Value *> TerminalValues;

std::map<std::string, std::string> HostSideKernelNameToOriginalNameMap;

std::map<Value *, bool> KernelLaunchIsIterative;
std::vector<Value *> KernelLaunches;

std::map<Instruction *, Instruction *> LIVTOInsertionPointMap;
std::map<Instruction*, Instruction*> KernelInvocationToInsertionPointMap;

    Instruction* FirstInvocation = nullptr;
    Instruction* FirstInvocationNonIter = nullptr;
// SCHostTransform
struct SCHostTransform : public ModulePass {
  static char ID; // Pass identification, replacement for typeid

  SCHostTransform() : ModulePass(ID) {}

  void processMemoryAllocation(CallBase *I) {
    errs() << "processing memory allocation\n";
    I->dump();
    I->getOperand(0)->dump();
    MallocPointers.insert(I->getOperand(0));
    // I->getOperand(1) ->dump();
    // I->getOperand(1) ->getType()->dump();
    if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // errs() << CI->getSExtValue() << "\n";
      MallocSizeMap[I] = CI->getSExtValue();
      /* MallocPointerToSizeMap[I->getOperand(0)] = CI->getSExtValue(); */
      auto OGPtr = PointerOpToOriginalPointers[I->getOperand(0)];
      if (OGPtr) {
        errs() << "og ptrs = ";
        OGPtr->dump();
        MallocPointerToSizeMap[OGPtr] = CI->getSExtValue();
        if (StructAllocas.find(OGPtr) != StructAllocas.end()) {
          errs() << "found struct og ptr\n";
          if (auto GEPI = dyn_cast<GetElementPtrInst>(I->getOperand(0))) {
            errs() << "found gepi\n";
            auto numIndices = GEPI->getNumIndices();
            if (numIndices == 2) {
              if (auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(2))) {
                errs() << "og is struct\n";
                PointerOpToOriginalStructPointersIndex[GEPI] =
                    FieldNum->getSExtValue();
                errs() << "field num = " << FieldNum->getSExtValue() << "\n";
                MallocPointerStructToIndexToSizeMap[OGPtr]
                                                   [FieldNum->getSExtValue()] =
                                                       CI->getSExtValue();
              }
            } else {
              if (auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(1))) {
                errs() << "og maybe struct or array\n";
                PointerOpToOriginalStructPointersIndex[GEPI] =
                    FieldNum->getSExtValue();
                errs() << "field num = " << FieldNum->getSExtValue() << "\n";
                MallocPointerStructToIndexToSizeMap[OGPtr]
                                                   [FieldNum->getSExtValue()] =
                                                       CI->getSExtValue();
              }
            }
          }
        }
      } else {
        if (FormalArgumentToActualArgumentMap.find(I->getOperand(0)) !=
            FormalArgumentToActualArgumentMap.end()) {
          errs() << "found actual arg\n";
          auto actarg = FormalArgumentToActualArgumentMap[I->getOperand(0)][0];
          actarg->dump();
          OGPtr = PointerOpToOriginalPointers[actarg];
          if (OGPtr) {
            errs() << "og ptrs = ";
            OGPtr->dump();
            MallocPointerToSizeMap[OGPtr] = CI->getSExtValue();
            if (StructAllocas.find(OGPtr) != StructAllocas.end()) {
              errs() << "found struct og ptr via args\n";
              /* MallocPointerStructToIndexToSizeMap[OGPtr][ */
            }
          }
        }
      }
    }
    else {
        MallocSizeMap[I->getOperand(0)] = 0;
            MallocPointerToSizeMap[I->getOperand(0)] = 0;
        MallocSizeMap[I] = 0;
            MallocPointerToSizeMap[I] = 0;
    }
    return;
  }

  int findKernelStructLocationForStoreInstruction(StoreInst *SI) {
    /* errs() << "STORE LOCATION TRACING\n"; */
    if (!SI) {
      /* errs() << "NOT A STORE\n"; */
      return -1;
    }
    /* SI->dump(); */
    // SI->getPointerOperand()->dump();
    if (GetElementPtrInst *GEPI =
            dyn_cast_or_null<GetElementPtrInst>(SI->getPointerOperand())) {
      /* errs() << "GEPI\n"; */
      /* GEPI->dump(); */
      /* errs() << "Pointer\n"; */
      /* GEPI->getPointerOperand()->dump(); */
      // errs() << GEPI->getNumIndices() << "\n";
      unsigned NumIndices = GEPI->getNumIndices();
      /* GEPI->getOperand(NumIndices)->dump(); */
      /* GEPI->getOperand(NumIndices)->getType()->dump(); */
      if (ConstantInt *CI =
              dyn_cast<ConstantInt>(GEPI->getOperand(NumIndices))) {
        return CI->getSExtValue();
      }
      errs() << "Unable to extract constant\n";
      return -1;
    }
    /* errs() << "Pointer\n"; */
    /* SI->getPointerOperand()->dump(); */
    return 0;
  }

  Value *recurseTillAllocation(Value *V) {
    V->dump();
    auto It = MallocSizeMap.find(V);
    if (It != MallocSizeMap.end()) {
      return V;
    }
    if (isa<llvm::PointerType>(V->getType())) {
      for (auto *user : V->users()) {
        if (isa<StoreInst>(user) && user->getOperand(1) == V) {
          return recurseTillAllocation(user);
        }
      }
    }
    if (isa<StoreInst>(V)) {
      auto *SI = dyn_cast<StoreInst>(V);
      if (isa<ConstantInt>(SI->getValueOperand())) {
        return SI->getValueOperand();
      }
      return recurseTillAllocation(SI->getPointerOperand());
    }
    return nullptr;
  }

  Value *findStoreInstOrStackCopyWithGivenValueOperand(Value *V) {
    errs() << "fsioscpwvo\n";
    for (auto *U : V->users()) {
      /* U->dump(); */
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V) {
          SI->dump();
          errs() << "store inst\n";
          return SI->getValueOperand();
        }
      }
      // also test for memcpy
      if (auto *CI = dyn_cast<CallBase>(U)) {
        // if  the passd value is the destination, then return the source
        auto Callee = CI->getCalledFunction();
        if ((Callee && (Callee->getName() == "llvm.memcpy.p0.p0.i64"))) {
          CI->dump();
          if (CI->getOperand(0) == V) {
            errs() << "memcpy call \n";
            return CI->getOperand(1);
          }
        }
      }
    }
    return nullptr;
  }

  StoreInst *findStoreInstWithGivenValueOperand(Value *V) {
    for (auto *U : V->users()) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getValueOperand() == V) {
          return SI;
        }
      }
    }
    return nullptr;
  }

  StoreInst *findStoreInstWithGivenPointerOperand(Value *V) {
    for (auto *U : V->users()) {
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V) {
          return SI;
        }
      }
    }
    return nullptr;
  }

  Value *findValueForStoreInstruction(StoreInst *SI) {
    /* errs() << "STORE VALUE TRACING\n"; */
    /* SI->getValueOperand()->dump(); */
    Value *ValueForStoreInst = SI->getValueOperand();
    // auto It = MallocSizeMap.find(ValueForStoreInst);
    // if (It != MallocSizeMap.end()) {
    return ValueForStoreInst;
    // }
    // return recurseTillAllocation(ValueForStoreInst);
  }

  void findAllocationOnLocalStack(CallBase* Invocation,
          Value* KernelArgStruct) {
      errs() << "findAllocationOnLocalStack\n";
      KernelArgStruct->dump();
      for (llvm::User *Karg : KernelArgStruct->users()) {
          errs() << "user: ";
          Karg->dump();
          StoreInst* KargSI = nullptr;
          /* if karg is store, direct store, else if karg is gep, look for store */
          if(auto* SI = dyn_cast<StoreInst>(Karg)) {
              /* errs() << "store inst\n"; */
              Value* Val = SI->getValueOperand();
              /* errs() << "storing "; */
              /* Val->dump(); */
              KargSI = SI;
          }
          // technically, there may be many geps before the store
          if(auto* GEPI = dyn_cast<GetElementPtrInst>(Karg)) {
              /* errs() << "gepi inst\n"; */
              // among users of this gepi, find the last store 
              for (llvm::User *GepiUser : GEPI->users()) {
                  /* GepiUser->dump(); */
                  if(auto* GESI = dyn_cast<StoreInst>(GepiUser)) {
                      /* errs() << "store inst\n"; */
                      Value* Val = GESI->getValueOperand();
                      /* errs() << "storing "; */
                      /* Val->dump(); */
                      KargSI = GESI;
                  }
              }
              /* errs() << "gepi user list over\n"; */
          }
          if(KargSI) {
              errs() << "user: kargsi\n";
              int Position = findKernelStructLocationForStoreInstruction(KargSI);
              Karg->dump();
              KargSI->dump();
              errs() << Position << "\n";
          } else {
              continue;
          }
          int Position = findKernelStructLocationForStoreInstruction(KargSI);
          errs() << "value stored in kargsi\n";
          KargSI->getValueOperand()->dump();
          // among its users, get the store
          for (llvm::User * KargSIUser: KargSI->getValueOperand()->users()) {
              if(auto* KargSIUserSI = dyn_cast<StoreInst>(KargSIUser)) {
                  if(KargSIUserSI->getPointerOperand() == KargSI->getValueOperand()) {
                      KargSIUser->dump();
                      KargSIUserSI->getValueOperand()->dump();
                      KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                          KargSIUserSI->getValueOperand();
                      KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                          KargSIUserSI->getValueOperand();
                      KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                          KargSIUserSI->getValueOperand();
                      KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] =
                          KargSIUserSI->getValueOperand();
                      errs() << "listing load all users of \n";
                      auto PtrLd = dyn_cast<LoadInst>(KargSIUserSI->getValueOperand());
                      if(PtrLd) {
                          auto Ptr = dyn_cast<AllocaInst>(PtrLd->getPointerOperand());
                          if(Ptr) {
                              Ptr->dump();
                              errs() << "users of now\n";
                              for (auto *user : Ptr->users()) {
                                  if(auto LDU = dyn_cast<LoadInst>(user)) {
                                      LDU->dump();
                                      AllocationToFirstMap[PtrLd] = LDU;
                                  }
                              }
                          }

                      }
                  }
                  if(KernelInvocationToEnclosingLIVMap.find(Invocation) != 
                          KernelInvocationToEnclosingLIVMap.end()) {
                  errs() << "match with LIV\n";
                      KernelInvocationToEnclosingLIVMap[Invocation]->dump();
                      if (KernelInvocationToEnclosingLIVMap[Invocation] ==
                              KargSIUserSI->getValueOperand()) {
                          errs() << "host loop\n";
                          Value *LIV = KargSIUserSI->getValueOperand();
                          LIV->dump();
                          KernelInvocationToArgNumberToLIVMap[Invocation][Position] = LIV;
                          KernelInvocationToLIVToArgNumMap[Invocation][LIV] = Position;
                          KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                              LIV;
                      }
                  }
              }
          }
          errs() << "end\n";
      }
  }

  // This function identifies the most recent store to the
  void recurseTillStoreOrEmtpy(CallBase *Invocation, Value *KernelArgStruct,
                               Value *V, Value *Karg) {
    /* errs() << "recurseTillStoreOrEmpty\n"; */
    V->dump();
    if (auto *SI = dyn_cast<StoreInst>(V)) {
      /* errs() << "STORE\n"; */
      KernelArgToStoreMap[KernelArgStruct].push_back(V);
      /* SI->dump(); */
      int Position = findKernelStructLocationForStoreInstruction(
          SI);                                       // store location tracing
      Value *Val = findValueForStoreInstruction(SI); // store value tracing
      errs() << "Position in Kernel Arg Struct = " << Position << "\n";
      if (Val) {
        errs() << "Value being written by store operand\n";
        Val->dump();
      }
      if (findStoreInstWithGivenValueOperand(
              Val)) { // find where the value is coming from
        errs() << "\nFOUND SIWGVO\n";
        findStoreInstWithGivenValueOperand(Val)->dump();
        auto *SIWGPO = findStoreInstWithGivenPointerOperand(Val);
        if (SIWGPO) {
          SIWGPO->dump();
          errs() << "SIWGPO value: ";
          SIWGPO->getValueOperand()->dump();
          if (FormalArgumentToActualArgumentMap.find(SIWGPO) !=
              FormalArgumentToActualArgumentMap.end()) {
            errs() << "found SIWGPO as an argument\n";
          }
          /* if (auto *LIPO = dyn_cast<LoadInst>(SIWGPO->getValueOperand())) { */
          /*   errs() << "\nWHICH USES LIPO\n"; */
          /*   LIPO->getPointerOperand()->dump(); */
          /*   auto MallocPointer = */
          /*       MallocPointerToSizeMap.find(LIPO->getPointerOperand()); */
          /*   if (MallocPointerToSizeMap.find(MallocPointer->first) != */
          /*       MallocPointerToSizeMap.end()) { */
          /*     errs() << "FOUND YAY!!\n"; */
          /*     /1* LIPO->getPointerOperand()->dump(); *1/ */
          /*     KernelInvocationToArgNumberToAllocationMap[Invocation][Position] = */
          /*         MallocPointer->first; */
          /*     KernelInvocationToArgNumberToActualArgMap[Invocation][Position] = */
          /*         MallocPointer->first; */
          /*     KernelInvocationToKernArgToAllocationMap[Invocation][Karg] = */
          /*         MallocPointer->first; */
          /*     KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] = */
          /*         SI; */
          /*     // LIPO->getPointerOperand(); */
          /*   } */
          /*   auto ArgumentPointer = LIPO->getPointerOperand(); */
          /*   if (FormalArgumentToActualArgumentMap.find(ArgumentPointer) != */
          /*       FormalArgumentToActualArgumentMap.end()) { */
          /*     errs() << "Found LIPO as a formal argument\n"; */
          /*     auto Ptr = FormalArgumentToActualArgumentMap[ArgumentPointer][0]; */
          /*     Ptr->dump(); */
          /*     if (PointerOpToOriginalPointers.find(Ptr) != */
          /*         PointerOpToOriginalPointers.end()) { */
          /*       errs() << "found OG pointer\n"; */
          /*       PointerOpToOriginalPointers[Ptr]->dump(); */
          /*       auto MallocPointer = PointerOpToOriginalPointers.find(Ptr); */
          /*       KernelInvocationToArgNumberToAllocationMap[Invocation] */
          /*                                                 [Position] = */
          /*                                                     MallocPointer */
          /*                                                         ->second; */
          /*       KernelInvocationToArgNumberToActualArgMap[Invocation] */
          /*                                                [Position] = */
          /*                                                    MallocPointer */
          /*                                                        ->second; */
          /*       KernelInvocationToKernArgToAllocationMap[Invocation][Karg] = */
          /*           MallocPointer->second; */
          /*       KernelInvocationToArgNumberToLastStoreMap[Invocation] */
          /*                                                [Position] = SI; */
          /*     } */
          /*   } */
          /* } */
          if (isa<PointerType>(SIWGPO->getValueOperand()->getType())) {
            errs() << "\n WHICH IS A POINTER\n";
            auto MallocPointer =
                PointerOpToOriginalPointers.find(SIWGPO->getValueOperand());
            if (MallocPointer != PointerOpToOriginalPointers.end()) {
              /* MallocPointer->first->dump(); */
              /* MallocPointer->second->dump(); */
              errs() << "FOUND YAY!!\n";
              /* LIPO->getPointerOperand()->dump(); */
              KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                  MallocPointer->first;
              KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                  MallocPointer->first;
              KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                  MallocPointer->first;
              KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] =
                  SI;
            }
          }
          if (auto *CIPO = dyn_cast<ConstantInt>(SIWGPO->getValueOperand())) {
            /* errs() << "constant\n"; */
            /* CIPO->dump(); */
            KernelInvocationToArgNumberToConstantMap[Invocation][Position] =
                CIPO;
            KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                CIPO;
          }
          if (KernelInvocationToEnclosingLIVMap[Invocation] ==
              SIWGPO->getValueOperand()) {
            errs() << "host loop\n";
            Value *LIV = SIWGPO->getValueOperand();
            LIV->dump();
            KernelInvocationToArgNumberToLIVMap[Invocation][Position] = LIV;
            KernelInvocationToLIVToArgNumMap[Invocation][LIV] = Position;
            KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                LIV;
          }
          KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
              SIWGPO->getValueOperand();
        } else {
          errs() << "complicated case\n";
          Val->dump();
          // iterate over all mempcy for structs and find out if any match Val
          if (MemcpyOpForStructsDstToInstMap.find(Val) !=
              MemcpyOpForStructsDstToInstMap.end()) {
            errs() << "found writer\n";
            auto memcpyInst = MemcpyOpForStructsDstToInstMap[Val];
            memcpyInst->dump();
            errs() << "source\n";
            auto src = memcpyInst->getOperand(1);
            src->dump();
              KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                  src;
              KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                  src;
              KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                  src;
              KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] =
                  SI;
            /* auto MallocPointer = PointerOpToOriginalPointers.find(src); */
            /* if (MallocPointer != PointerOpToOriginalPointers.end()) { */
            /*   errs() << "original pointer\n"; */
            /*   MallocPointer->second->dump(); */
            /*   KernelInvocationToArgNumberToAllocationMap[Invocation][Position] = */
            /*       MallocPointer->second; */
            /*   KernelInvocationToArgNumberToActualArgMap[Invocation][Position] = */
            /*       MallocPointer->second; */
            /*   KernelInvocationToKernArgToAllocationMap[Invocation][Karg] = */
            /*       MallocPointer->second; */
            /*   KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] = */
            /*       SI; */
            /* } */
          }
        }
      }
      return;
    }
    for (auto *U : V->users()) {
      recurseTillStoreOrEmtpy(Invocation, KernelArgStruct, U, Karg);
    }
    return;
  }

  std::string getOriginalKernelName(std::string Mangledname) {
    /* errs() << "name check: " << Mangledname << "\n"; */
    /* errs() << "host side name : " <<
     * HostSideKernelNameToOriginalNameMap[Mangledname] << "\n"; */
    return HostSideKernelNameToOriginalNameMap[Mangledname];
    /* return Mangledname.erase(0, 19); */
  }

  bool isNumber(std::string op) {
    bool isNum = true;
    for (int i = 0; i < op.length(); i++) {
      if (op[0] == '-') {
        continue;
      }
      if (!isDigit(op[i])) {
        isNum = false;
        break;
      }
    }
    return isNum;
  }

  ExprTreeOp getExprTreeOp(std::string op) {
    if (isNumber(op)) {
      return ETO_CONST;
    }
    if (op.compare("PC") == 0) {
      return ETO_PC;
    } else if (op.compare("ADD") == 0) {
      return ETO_ADD;
    } else if (op.compare("SUB") == 0) {
      return ETO_SUB;
    } else if (op.compare("AND") == 0) {
      return ETO_AND;
    } else if (op.compare("OR") == 0) {
      return ETO_OR;
    } else if (op.compare("MUL") == 0) {
      return ETO_MUL;
    } else if (op.compare("SHL") == 0) {
      return ETO_SHL;
    } else if (op.compare("LSHR") == 0) {
      return ETO_LSHR;
    } else if (op.compare("DIV") == 0) {
      return ETO_DIV;
    } else if (op.compare("UDIV") == 0) {
      return ETO_UDIV;
    } else if (op.compare("SDIV") == 0) {
      return ETO_SDIV;
    } else if (op.compare("SREM") == 0) {
      return ETO_SREM;
    } else if (op.compare("FDIV") == 0) {
      return ETO_FDIV;
    } else if (op.compare("FMUL") == 0) {
      return ETO_FMUL;
    } else if (op.compare("PHI") == 0) {
      return ETO_PHI;
    } else if (op.compare("ICMP") == 0) {
      return ETO_ICMP;
    } else if (op.compare("FCMP") == 0) {
      return ETO_FCMP;
    } else if (op.compare("LOAD") == 0) {
      return ETO_MEMOP;
    } else if (op.compare("STORE") == 0) {
      return ETO_MEMOP;
    } else if (op.compare("TIDX") == 0) {
      return ETO_TIDX;
    } else if (op.compare("TIDY") == 0) {
      return ETO_TIDY;
    } else if (op.compare("BIDX") == 0) {
      return ETO_BIDX;
    } else if (op.compare("BIDY") == 0) {
      return ETO_BIDY;
    } else if (op.compare("BDIMX") == 0) {
      return ETO_BDIMX;
    } else if (op.compare("BDIMY") == 0) {
      return ETO_BDIMY;
    } else if (op.compare("GEP") == 0) {
      return ETO_GEP;
    } else if (op.compare("ZEXT") == 0) {
      return ETO_ZEXT;
    } else if (op.compare("SEXT") == 0) {
      return ETO_SEXT;
    } else if (op.compare("FREEZE") == 0) {
      return ETO_FREEZE;
    } else if (op.compare("double") == 0) {
      return ETO_DOUBLE;
    } else if (op.compare("TRUNC") == 0) {
      return ETO_TRUNC;
    } else if (op.compare("FPTOSI") == 0) {
      return ETO_FPTOSI;
    } else if (op.compare("SITOFP") == 0) {
      return ETO_SITOFP;
    } else if (op.compare("UITOFP") == 0) {
      return ETO_UITOFP;
    } else if (op.compare("SELECT") == 0) {
      return ETO_SELECT;
    } else if (op.compare("CALL") == 0) {
      return ETO_CALL;
    } else if (op.compare("ATOMICRMW") == 0) {
      return ETO_ATOMICRMW;
    } else if (op.compare("UNDEF") == 0) {
      return ETO_UNDEF;
    } else if (op.compare("LOAD") == 0) {
      return ETO_LOAD;
    } else if (op.compare("INCOMP") == 0) {
      return ETO_INCOMP;
    } else if (op.compare("UNKNOWN") == 0) {
      return ETO_UNKNOWN;
    } else if (op.substr(0, 3).compare("ARG") == 0) {
      return ETO_ARG;
    } else if (op.substr(0, 3).compare("PHI") == 0) {
      return ETO_PHI;
    }
    assert(0);
    return ETO_NONE;
  }

  unsigned getExprTreeNodeArg(std::string op) { return stoi(op.substr(3)); }
  unsigned getExprTreePhiArg(std::string op) { return stoi(op.substr(3)); }

  bool isTerminal(ExprTreeNode *node) {
    /* errs() << node->op << "\n"; */
    if (terminals.find(node->op) != terminals.end()) {
      return true;
    }
    return false;
  }

  bool isTerminal(ExprTreeNodeAdvanced *node) {
    /* errs() << node->op << "\n"; */
    if (terminals.find(node->op) != terminals.end()) {
      return true;
    }
    return false;
  }

  bool isPhiNode(ExprTreeNode *node) {
    if (node->op == ETO_PHI)
      return true;
    return false;
  }

  bool isOperation(ExprTreeNode *node) {
    if (operations.find(node->op) != operations.end()) {
      return true;
    }
    return false;
  }

  bool isOperation(ExprTreeNodeAdvanced *node) {
    if (operations.find(node->op) != operations.end()) {
      return true;
    }
    return false;
  }

  bool detectParticularNode(ExprTreeNodeAdvanced* root, ExprTreeOp op) {
      if (root == nullptr) {
          return false;
      }
      /* errs() << "\ntraverse expression tree\n"; */
      std::stack<ExprTreeNodeAdvanced *> Stack;
      Stack.push(root);
      while (!Stack.empty()) {
          ExprTreeNodeAdvanced *Current = Stack.top();
          errs() << Current->original_str << " ";
          if(Current->op == op) {
              return true;
          }
          Stack.pop();
          /* if (isOperation(Current)) { */
              for(auto child = Current->children.begin(); child != Current->children.end(); child++) {
                  Stack.push(*child);
              }
          /* } */
      }
      return false;
  }

  void traverseExpressionTree(ExprTreeNodeAdvanced *root) {
      if (root == nullptr)
          return;
      /* errs() << "\ntraverse expression tree\n"; */
      std::stack<ExprTreeNodeAdvanced *> Stack;
      Stack.push(root);
      while (!Stack.empty()) {
          ExprTreeNodeAdvanced *Current = Stack.top();
          errs() << Current->original_str << " ";
          Stack.pop();
          /* if (isOperation(Current)) { */
              for(auto child = Current->children.begin(); child != Current->children.end(); child++) {
                  Stack.push(*child);
              }
          /* } */
      }
  }

  void traverseExpressionTree(ExprTreeNode *root) {
    if (root == nullptr)
      return;
    /* errs() << "\ntraverse expression tree\n"; */
    std::stack<ExprTreeNode *> Stack;
    Stack.push(root);
    while (!Stack.empty()) {
      ExprTreeNode *Current = Stack.top();
      errs() << Current->original_str << " ";
      Stack.pop();
      if (isOperation(Current)) {
        Stack.push(Current->children[0]);
        Stack.push(Current->children[1]);
      }
    }
  }

  ExprTreeNode *findNodeInExpressionTree(ExprTreeNode *root, ExprTreeOp op,
                                         unsigned arg) {
    /* errs() << "\nfind node in expression tree\n"; */
    std::stack<ExprTreeNode *> Stack;
    Stack.push(root);
    while (!Stack.empty()) {
      ExprTreeNode *Current = Stack.top();
      if (Current->op == op) {
        if (op == ETO_ARG) {
          if (Current->arg == arg) {
            return Current;
          }
        } else {
          return Current;
        }
      }
      /* errs() << Current->original_str << " "; */
      Stack.pop();
      if (isOperation(Current)) {
        Stack.push(Current->children[0]);
        Stack.push(Current->children[1]);
      }
    }
    return nullptr;
  }

  unsigned getMaxValueForLiterals(CallBase *CI, ExprTreeNode *node,
                                  unsigned LoopArg, unsigned loopid) {
    if (node->op == ETO_ARG) {
      if (node->arg == LoopArg) {
        // TODO:  get loop bounds
        return 0;
      } else {
        return getActualHostValueForLiterals(CI, node);
      }
    }
    if (node->op == ETO_PHI_TERM) {
      // get lower bound of phi terminal from loop information file
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      if (loopid == 0) {
        return 1;
      }
      auto Fin = LoopIDToBoundsExprMapFin[OriginalKernelName][loopid];
      return evaluateExpressionTree(CI, Fin);
    }
    if (node->op == ETO_BIDX || node->op == ETO_BIDY || node->op == ETO_TIDX ||
        node->op == ETO_TIDY) {
      if (node->op == ETO_TIDX) {
        return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX] - 1;
      }
      if (node->op == ETO_TIDY) {
        return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY] - 1;
      }
      if (node->op == ETO_BIDX) {
        if (KernelInvocationToGridSizeValueMap[CI].find(AXIS_TYPE_GDIMX) !=
            KernelInvocationToGridSizeValueMap[CI].end()) {
          auto GridValueSize =
              KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
          auto RRPN = getExpressionTree(GridValueSize);
          auto griddimx = evaluateRPNForIter0(CI, RRPN);
          return griddimx;
        } else {
          errs() << "hehe: "
                 << KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX] << "\n";
          return KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX] - 1;
        }
      }
      if (node->op == ETO_BIDY) {
        if (KernelInvocationToGridSizeValueMap[CI].find(AXIS_TYPE_GDIMY) !=
            KernelInvocationToGridSizeValueMap[CI].end()) {
          auto GridValueSize =
              KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY];
          auto RRPN = getExpressionTree(GridValueSize);
          auto griddimy = evaluateRPNForIter0(CI, RRPN);
          return griddimy;
        } else {
          return KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY] - 1;
        }
      }
    }
    return getActualHostValueForLiterals(CI, node);
  }

  unsigned getMinValueForLiterals(CallBase *CI, ExprTreeNode *node,
                                  unsigned LoopArg, unsigned loopid) {
    if (node->op == ETO_ARG) {
      if (node->arg == LoopArg) {
        // TODO:  get loop bounds
        return 0;
      } else {
        return getActualHostValueForLiterals(CI, node);
      }
    }
    if (node->op == ETO_PHI_TERM) {
      // get lower bound of phi terminal from loop information file
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      if (loopid == 0) {
        return 1;
      }
      auto In = LoopIDToBoundsExprMapIn[OriginalKernelName][loopid];
      return evaluateExpressionTree(CI, In);
    }
    if (node->op == ETO_BIDX || node->op == ETO_BIDY || node->op == ETO_TIDX ||
        node->op == ETO_TIDY) {
      return 0;
    }
    return getActualHostValueForLiterals(CI, node);
  }

  unsigned getActualHostValueForLiterals(CallBase *CI, ExprTreeNode *node) {
    if (node->op == ETO_INTERM) {
      return node->value;
    }
    if (node->op == ETO_CONST) {
      return stoi(node->original_str);
    }
    if (node->op == ETO_BDIMX) {
      return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
    }
    if (node->op == ETO_BDIMY) {
      return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
    }
    if (node->op == ETO_BIDX || node->op == ETO_BIDY) {
      return 1;
    }
    if (node->op == ETO_ARG) {
      std::map<unsigned, Value *> ArgNumToConstantMap =
          KernelInvocationToArgNumberToConstantMap[CI];
      if (ArgNumToConstantMap.find(node->arg) != ArgNumToConstantMap.end()) {
        Value *ConstArg = ArgNumToConstantMap[node->arg];
        if (ConstantInt *CI = dyn_cast<ConstantInt>(ConstArg)) {
          return CI->getSExtValue();
        }
        return 0;
      }
      // here is to add constant arguments from host side used as arguments to
      // kernel side
      Value *ArgInQ = KernelInvocationToArgNumberToActualArgMap[CI][node->arg];
      if (ArgInQ) {
        /* CI->dump(); */
        /* errs() << "ARG IN Q\n"; */
        /* ArgInQ->dump(); */
        /* errs() << "actual arg\n"; */
        /* FunctionCallToFormalArgumentToActualArgumentMap[CI][ArgInQ]->dump();
         */
        // we are assuming there is only one call to each function containing a
        // kernel invocation.
        /* FormalArgumentToActualArgumentMap[ArgInQ][0]->dump(); */
        if (auto ConstI = dyn_cast<ConstantInt>(
                FormalArgumentToActualArgumentMap[ArgInQ][0])) {
          return ConstI->getSExtValue();
        }
        // MAJOR TODO: ArgInQ may have to be traversed in reverse till it
        // becomes:function arg
      }
    }
    // so on and so forth
    // get constant args
    return 0;
  }

  ExprTreeNode *operateMax(CallBase *CI, ExprTreeNode *operation,
                           ExprTreeNode *op1, ExprTreeNode *op2,
                           unsigned LoopArg, unsigned loopid) {
    ExprTreeNode *result = new ExprTreeNode();
    unsigned long long v1 = getMaxValueForLiterals(CI, op1, LoopArg, loopid);
    unsigned long long v2 = getMaxValueForLiterals(CI, op2, LoopArg, loopid);
    unsigned long long res = 1;
    errs() << operation->original_str << "::::" << v1 << " " << v2 << "\n";
    if (operation->op == ETO_SHL) {
      res = v1 << v2;
    }
    if (operation->op == ETO_MUL) {
      res = v1 * v2;
    }
    if (operation->op == ETO_ADD) {
      res = v1 + v2;
    }
    if (operation->op == ETO_OR) {
      res = v1 + v2;
    }
    if (operation->op == ETO_PHI) {
      res = (v1 < v2) ? v2 : v1;
    }
    result->op = ETO_INTERM;
    result->value = res;
    return result;
  }

  ExprTreeNode *operateMin(CallBase *CI, ExprTreeNode *operation,
                           ExprTreeNode *op1, ExprTreeNode *op2,
                           unsigned LoopArg, unsigned loopid) {
    ExprTreeNode *result = new ExprTreeNode();
    unsigned long long v1 = getMinValueForLiterals(CI, op1, LoopArg, loopid);
    unsigned long long v2 = getMinValueForLiterals(CI, op2, LoopArg, loopid);
    unsigned long long res = 1;
    /* errs() << operation->original_str << "::::" << v1 << " " << v2 << "\n";
     */
    if (operation->op == ETO_SHL) {
      res = v1 << v2;
    }
    if (operation->op == ETO_MUL) {
      res = v1 * v2;
    }
    if (operation->op == ETO_ADD) {
      res = v1 + v2;
    }
    if (operation->op == ETO_OR) {
      res = v1 + v2;
    }
    if (operation->op == ETO_PHI) {
      res = (v1 < v2) ? v1 : v2;
    }
    result->op = ETO_INTERM;
    result->value = res;
    return result;
  }

  ExprTreeNode *operate(CallBase *CI, ExprTreeNode *operation,
                        ExprTreeNode *op1, ExprTreeNode *op2) {
    ExprTreeNode *result = new ExprTreeNode();
    unsigned long long v1 = getActualHostValueForLiterals(CI, op1);
    unsigned long long v2 = getActualHostValueForLiterals(CI, op2);
    unsigned long long res = 1;
    /* errs() << v1 << " " << v2 << "\n"; */
    if (operation->op == ETO_SHL) {
      res = v1 << v2;
    }
    if (operation->op == ETO_MUL) {
      res = v1 * v2;
    }
    if (operation->op == ETO_ADD) {
      res = v1 + v2;
    }
    if (operation->op == ETO_OR) {
      res = v1 + v2;
    }
    result->op = ETO_INTERM;
    result->value = res;
    return result;
  }

  unsigned long long evaluateRPNforMax(CallBase *CI,
                                       std::vector<ExprTreeNode *> RPN,
                                       unsigned LoopArg, unsigned loopid) {
    errs() << "Evaluating RPN for max\n";
    std::stack<ExprTreeNode *> stack;
    for (auto Token = RPN.begin(); Token != RPN.end(); Token++) {
      errs() << (*Token)->original_str << "\n ";
      if (isOperation(*Token)) {
        /* errs() << "operation\n"; */
        ExprTreeNode *op1 = stack.top();
        stack.pop();
        ExprTreeNode *op2 = stack.top();
        stack.pop();
        // evaluate the expression
        ExprTreeNode *result;
        if (isTerminal(op1) && isTerminal(op2)) {
          result = operateMax(CI, *Token, op1, op2, LoopArg, loopid);
          errs() << "interm = " << result->value << "\n";
        } else {
          errs() << "MAJOR ISSUE: node not teminal\n";
        }
        result->op = ETO_INTERM;
        stack.push(result);
        /* errs() << "TOS = " << result->value << "\n"; */
      } else {
        /* errs() << "terminal\n"; */
        (*Token)->value = getMaxValueForLiterals(CI, (*Token), LoopArg, loopid);
        stack.push((*Token));
      }
    }
    return stack.top()->value;
  }

  unsigned long long evaluateRPNforMin(CallBase *CI,
                                       std::vector<ExprTreeNode *> RPN,
                                       unsigned LoopArg, unsigned loopid) {
    /* errs() << "Evaluating RPN for min\n"; */
    std::stack<ExprTreeNode *> stack;
    for (auto Token = RPN.begin(); Token != RPN.end(); Token++) {
      /* errs() << (*Token)->original_str << ": "; */
      if (isOperation(*Token)) {
        /* errs() << "operation\n"; */
        ExprTreeNode *op1 = stack.top();
        stack.pop();
        ExprTreeNode *op2 = stack.top();
        stack.pop();
        // evaluate the expression
        ExprTreeNode *result;
        if (isTerminal(op1) && isTerminal(op2)) {
          result = operateMin(CI, *Token, op1, op2, LoopArg, loopid);
          /* errs() << "interm = " << result->value << "\n"; */
        } else {
          errs() << "MAJOR ISSUE: node not teminal\n";
        }
        result->op = ETO_INTERM;
        stack.push(result);
        /* errs() << "TOS = " << result->value << "\n"; */
      } else {
        /* errs() << "terminal\n"; */
        (*Token)->value = getMinValueForLiterals(CI, (*Token), LoopArg, loopid);
        stack.push((*Token));
      }
    }
    return stack.top()->value;
  }

  unsigned long long evaluateRPN(CallBase *CI,
                                 std::vector<ExprTreeNode *> RPN) {
    /* errs() << "Evaluating RPN\n"; */
    std::stack<ExprTreeNode *> stack;
    for (auto Token = RPN.begin(); Token != RPN.end(); Token++) {
      /* errs() << (*Token)->original_str << ": "; */
      if (isOperation(*Token)) {
        /* errs() << "operation\n"; */
        ExprTreeNode *op1 = stack.top();
        stack.pop();
        ExprTreeNode *op2 = stack.top();
        stack.pop();
        // evaluate the expression
        ExprTreeNode *result;
        if (isTerminal(op1) && isTerminal(op2)) {
          result = operate(CI, *Token, op1, op2);
          /* errs() << "interm = " << result->value << "\n"; */
        } else {
          errs() << "MAJOR ISSUE: node not teminal\n";
        }
        result->op = ETO_INTERM;
        stack.push(result);
      } else {
        /* errs() << "terminal\n"; */
        (*Token)->value = getActualHostValueForLiterals(CI, (*Token));
        stack.push((*Token));
      }
    }
    return stack.top()->value;
  }

  // substitute values from host side to get concrete values
  unsigned long long evaluateExpressionTree(CallBase *CI, ExprTreeNode *root) {
    /* errs() << "evaluating expr rooted at " << root->original_str << "\n"; */
    // Convert to RPN
    std::stack<ExprTreeNode *> Stack;
    std::vector<ExprTreeNode *> RPN;
    Stack.push(root);
    while (!Stack.empty()) {
      ExprTreeNode *Current = Stack.top();
      /* errs() << Current->original_str << " "; */
      Stack.pop();
      RPN.push_back(Current);
      if (isOperation(Current)) {
        Stack.push(Current->children[0]);
        Stack.push(Current->children[1]);
      }
    }
    reverse(RPN.begin(), RPN.end());
    // Evaluate RPN
    unsigned long long value = evaluateRPN(CI, RPN);
    /* errs() << "\n eval over = " << value << "\n"; */
    return value;
  }

  std::vector<ExprTreeNode *>
  findMultipliersByTraversingUpExprTree(ExprTreeNode *root,
                                        ExprTreeNode *given) {
    errs() << "\nfind multipliers\n";
    std::vector<ExprTreeNode *> Multipliers;
    ExprTreeNode *current = given;
    ExprTreeNode *parent = current->parent;
    while (current != nullptr) {
      parent = current->parent;
      if (parent) {
          errs()  << parent->original_str << "\n";
        if (parent->op == ETO_MUL || parent->op == ETO_SHL) {
          errs() << parent->original_str << "  " << parent->op << "\n";
          // URGENT TODO: if op is shl, then we need to note that somehow
          if (parent->children[0] == current) {
            Multipliers.push_back(parent->children[1]);
          } else {
            Multipliers.push_back(parent->children[0]);
          }
          errs() << "pushed to Multipliers\n";
        }
      }
      current = parent;
    }
    return Multipliers;
  }

  std::vector<ExprTreeNode *>
  findDivisorsByTraversingUpExprTree(ExprTreeNode *root,
                                        ExprTreeNode *given) {
    errs() << "\nfind divisors\n";
    std::vector<ExprTreeNode *> Multipliers;
    ExprTreeNode *current = given;
    ExprTreeNode *parent = current->parent;
    while (current != nullptr) {
      parent = current->parent;
      if (parent) {
        if (parent->op == ETO_UDIV || parent->op == ETO_SDIV) {
          errs() << parent->original_str << "  " << parent->op << "\n";
          // URGENT TODO: if op is shl, then we need to note that somehow
          if (parent->children[0] == current) {
            Multipliers.push_back(parent->children[1]);
          } else {
            Multipliers.push_back(parent->children[0]);
          }
        }
      }
      current = parent;
    }
    return Multipliers;
  }

  // We only work when PHI nodes have two incoming paths.
  // Further, we only handle cases where the values across iteratins are of the
  // form C*i, where i is the induction and C is a
  unsigned long long partialDifferenceWRTPhi(CallBase *CI, ExprTreeNode *root) {
    errs() << "\npartial diff with rt phi\n";
    ExprTreeNode *phi = findNodeInExpressionTree(root, ETO_PHI, 0);
    ExprTreeNode *phiterm = findNodeInExpressionTree(root, ETO_PHI_TERM, 0);
    std::vector<ExprTreeNode *> Adders;
    ExprTreeNode *current = phiterm;
    ExprTreeNode *parent;
    if (phi && phiterm) {
      errs() << phi->original_str << "\n";
      errs() << phiterm->original_str << "\n";
      while (current != phi) {
        parent = current->parent;
        if (parent->children[0] == current) {
          Adders.push_back(parent->children[1]);
        }
        if (parent->children[1] == current) {
          Adders.push_back(parent->children[0]);
        }
        current = parent;
      }
      errs() << "partial difference wrt phi node\n";
      unsigned long long partialDiffWRTPhi =
          evaluateExpressionTree(CI, Adders[0]);
      return partialDiffWRTPhi;
    }
    return 0;
  }

  unsigned partialDifferenceOfExpressionTreeWRTGivenNode(CallBase *CI,
                                                         ExprTreeNode *root,
                                                         ExprTreeOp given,
                                                         unsigned arg) {
    // Find the given op in the expression tree
    // TODO: fin all nodes of the given type in the expression tree
    ExprTreeNode *node = findNodeInExpressionTree(root, given, arg);
    if (node)
      errs() << "found node " << node->original_str << "\n";
    else
      errs() << "not found node \n";
    // move up towards root, looking for MUL/SHLs. Store the other child.
    if (node) {
      auto mutlipliers = findMultipliersByTraversingUpExprTree(root, node);
      errs() << "multipliers => ";
      for (auto mutliplier = mutlipliers.begin();
           mutliplier != mutlipliers.end(); mutliplier++) {
        errs() << (*mutliplier)->original_str << ".";
      }

      // evaluate the stored other childs and multiply all of them together
      unsigned long long FinalMultiplier = 1;
      for (auto mutliplier = mutlipliers.begin();
           mutliplier != mutlipliers.end(); mutliplier++) {
        if ((*mutliplier)->parent->op == ETO_MUL) {
          FinalMultiplier *= evaluateExpressionTree(CI, (*mutliplier));
          errs() << "finmul = " << FinalMultiplier << "\n";
        }
        if ((*mutliplier)->parent->op == ETO_SHL) {
          FinalMultiplier = FinalMultiplier
                            << evaluateExpressionTree(CI, (*mutliplier));
          errs() << "finmul = " << FinalMultiplier << "\n";
        }
      }
      return FinalMultiplier;
    }

    return 0;
  }

  ExprTreeNode *createExpressionTree(std::vector<std::string> RPN) {
    ExprTreeNode *root = nullptr;
    ExprTreeNode *current = nullptr;
    std::stack<ExprTreeNode *> stack;
    std::vector<ExprTreeNode *> RPN_Nodes;
    /* errs() << "\ncreate expression tree\n"; */
    reverse(RPN.begin(), RPN.end());
    if (RPN.size() == 0) {
      return nullptr;
    }
    if(RPN.size() > 50) {
      current = new ExprTreeNode();
      current->op = ETO_PC; // TODO: use a different technique
      return current;
    }
    if(RPN[0].compare("INCOMP") == 0) {
      current = new ExprTreeNode();
      current->op = ETO_INCOMP;
      return current;
    }
    for (auto str = RPN.begin(); str != RPN.end(); str++) {
      /* errs() << *str << "\n"; */
      current = new ExprTreeNode();
      current->op = getExprTreeOp(*str);
      current->original_str = *str;
      current->parent = nullptr;
      RPN_Nodes.push_back(current);
    }
    for (auto node = RPN_Nodes.begin(); node != RPN_Nodes.end(); node++) {
      if ((*node)->op == ETO_ARG) {
        /* errs() << (*node)->original_str << " " ; */
        (*node)->arg = getExprTreeNodeArg((*node)->original_str);
        /* errs() << (*node)->arg << " "; */
      }
    }
    // make tree using stack
    /* errs() << "\nmaking tree\n"; */
    bool phi_term_seen = false;
    for (auto node = RPN_Nodes.begin(); node != RPN_Nodes.end(); node++) {
      errs() << "\n" << (*node)->original_str << " ";
      if (isPhiNode(*node)) { // first phi node is term, second and later are
                              // ops: FIXME
        if (phi_term_seen == false) {
          errs() << "Terminal PHI";
          (*node)->op = ETO_PHI_TERM;
          stack.push(*node);
          phi_term_seen = true;
        } else {
          errs() << "Operation PHI";
          if (stack.empty())
            return nullptr;
          ExprTreeNode *child1 = stack.top();
          stack.pop();
          if (stack.empty())
            return nullptr;
          ExprTreeNode *child2 = stack.top();
          stack.pop();
          child1->parent = *node;
          child2->parent = *node;
          (*node)->children[0] = child1;
          (*node)->children[1] = child2;
          stack.push(*node);
          phi_term_seen = false; // Will this work properly?
        }
      } else if (isOperation(*node)) {
        errs() << "Operation ";
        if (stack.empty())
          return nullptr;
        ExprTreeNode *child1 = stack.top();
        stack.pop();
        if (stack.empty())
          return nullptr;
        ExprTreeNode *child2 = stack.top();
        stack.pop();
        child1->parent = *node;
        child2->parent = *node;
        (*node)->children[0] = child1;
        (*node)->children[1] = child2;
        stack.push(*node);
      } else { // push to stack
        stack.push(*node);
      }
    }
    root = stack.top();
    stack.pop();
    traverseExpressionTree(root);
    return root;
  }

  // Create expression tree from parenthesised serialized expression tree
  ExprTreeNodeAdvanced* createExpressionTreeAdvanced(std::vector<std::string> serializedTree){
      ExprTreeNodeAdvanced *root = nullptr;
      ExprTreeNodeAdvanced *current = nullptr;
      std::stack<ExprTreeNodeAdvanced *> stack;
      unsigned term_count = 0;
      for (auto str = serializedTree.begin(); str != serializedTree.end(); str++) {
          errs()<<"adv expr tree " << *str << "\n";
          if(*str == "(" ) {
              str++;
          errs()<<"adv expr tree " << *str << "\n";
              ExprTreeNodeAdvanced* node = new ExprTreeNodeAdvanced();
              node->op = getExprTreeOp(*str);
              node->original_str = *str;
              node->parent = nullptr;
              term_count++;
              if(term_count > 100) {
                  break;
              }
              if(node->op == ETO_ARG) {
                  node->arg = getExprTreeNodeArg(*str);
              }
              if(node->op == ETO_PHI) {
                  node->arg = getExprTreePhiArg(*str);
              }
              if(root == nullptr) {
                  root = node;
              } else {
                 stack.top()->children.push_back(node);
                 node->parent = stack.top();
              }
              stack.push(node);
          } else if(*str == ")" ) {
              current = stack.top();
              stack.pop();
          }
      }
      return root;
  }

  void printExpresstionTreeAdvanced(ExprTreeNodeAdvanced* root) {
      if (!root) return;
      errs() << root->original_str << " ";
      errs() << "( ";
      for (ExprTreeNodeAdvanced* child : root->children) {
          printExpresstionTreeAdvanced(child);
      }
      errs() << ") ";
  }

  void printKernelDeviceAnalyis() {
    std::string Data;

    std::ifstream LoopDetailFile("loop_detail_file.lst");
    if (LoopDetailFile.is_open()) {
      errs() << "Reading Loop Detail File\n";
      std::string line;
      while (getline(LoopDetailFile, line)) {
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        /* word.erase(0,4); */
        std::string KernelName = word;
        errs() << KernelName << " ";
        ss >> word;
        unsigned LoopId = stoi(word);
        ss >> word;
        unsigned ParentLoopId = stoi(word);
        /* errs() << LoopId << " "; */
        LoopIDToParentLoopIDMap[LoopId] = ParentLoopId;
        std::vector<std::string> IN, FIN, STEP;
        int in = 0, fin = 0, step = 0, iters = 0;
        while (ss >> word) {
          /* errs() << word << " "; */
          if (word.compare("IT") == 0) {
            ss >> word;
            iters = stoi(word);
            /* errs() << iters; */
            LoopIDToLoopItersMap[KernelName][LoopId] = iters;
          } else {
            /* errs() << word << " "; */
            LoopIDToLoopBoundsMap[KernelName][LoopId].push_back(word);
          }
        }
        errs() << "\n";
      }
    }

    std::ifstream PhiLoopFile("phi_loop_file.lst");
    if(PhiLoopFile.is_open()) {
      errs() << "Reading PHI loop File\n";
      std::string line;
      while (getline(PhiLoopFile, line)) {
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        unsigned PhiID = stoi(word);
        ss >> word;
        unsigned LoopID = stoi(word);
        PhiNodeToLoopIDMap[PhiID] = LoopID;
      }
    }

    std::ifstream IfDetailFile("if_detail_file.lst");
    if (IfDetailFile.is_open()) {
      errs() << "Reading If Detail File\n";
      std::string line;
      while (getline(IfDetailFile, line)) {
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        unsigned IfId = stoi(word);
        errs() << IfId << " ";
        while (ss >> word) {
          errs() << word << " ";
          IfIDToCondMap[IfId].push_back(word);
        }
        errs() << "\n";
      }
    }

    std::ifstream AccessDetailFile("access_detail_file.lst");
    if (AccessDetailFile.is_open()) {
      errs() << "Reading Access Detail File\n";
      std::string line;
      while (getline(AccessDetailFile, line)) {
        /* errs() << line; */
        /* errs() << "\n"; */
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        /* word.erase(0,4); */
        std::string KernelName = word;
        errs() << KernelName << " ";
        ss >> word;
        unsigned AccessId = stoi(word);
        errs() << AccessId << " ";
        ss >> word;
        unsigned ParamNumber = stoi(word);
        errs() << ParamNumber << " ";
        KernelNameToAccessIDToAllocationArgMap[KernelName][AccessId] =
            ParamNumber;
        ss >> word;
        unsigned LoopId = stoi(word);
        errs() << LoopId << " ";
        KernelNameToAccessIDToEnclosingLoopMap[KernelName][AccessId] = LoopId;
        ss >> word;
        unsigned IfId = stoi(word);
        errs() << IfId << " ";
        KernelNameToAccessIDToIfCondMap[KernelName][AccessId] = IfId;
        ss >> word;
        unsigned IfType = stoi(word);
        errs() << IfType << " ";
        KernelNameToAccessIDToIfTypeMap[KernelName][AccessId] = IfType;
        std::vector<std::string> RPN;
        while (ss >> word) {
          errs() << word << " ";
          if (word.compare("[") == 0 || word.compare("]") == 0) {
            ;
          } else {
            RPN.push_back(word);
          }
        }
        errs() << "good expression\n";
        KernelNameToAccessIDToExpressionTreeMap[KernelName][AccessId] =
            createExpressionTree(RPN);
        errs() << "\n";
      }
    }

    std::ifstream AccessTreeFile("access_tree_file.lst");
    if (AccessTreeFile.is_open()) {
      errs() << "Reading Access Tree File\n";
      std::string line;
      while (getline(AccessTreeFile, line)) {
        /* errs() << line; */
        /* errs() << "\n"; */
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        /* word.erase(0,4); */
        std::string KernelName = word;
        errs() << KernelName << " ";
        ss >> word;
        unsigned AccessId = stoi(word);
        errs() << AccessId << " ";
        std::vector<std::string> RPN;
        while (ss >> word) {
          errs() << word << " ";
          if (word.compare("[") == 0 || word.compare("]") == 0) {
            ;
          } else {
            RPN.push_back(word);
          }
        }
        errs() << "good expression\n";
        auto test = createExpressionTreeAdvanced(RPN);
        printExpresstionTreeAdvanced(test);
        KernelNameToAccessIDToAdvancedExpressionTreeMap[KernelName][AccessId] =
            test;
        errs() << "\n";
      }
    }



    /* std::ifstream AccessDetailFile("access_detail_file.lst"); */
    /* errs() << "DEVICE ANALYSIS\n"; */
    /* while (AccessDetailFile >> Data) { */
    /*   errs() << Data << "\n"; */
    /*   Data.erase(0, 4); */
    /*   std::string KernelName = Data; */
    /*   // errs() << Data << "\n"; // need to unmangle somehow! */
    /*   AccessDetailFile >> Data; */
    /*   errs() << Data << "\n"; */
    /*   unsigned ParamNumber = stoi(Data); */
    /*   // errs() << Data << "\n"; */
    /*   AccessDetailFile >> Data; */
    /*   errs() << Data << "\n"; */
    /*   unsigned AccessInfo = stoi(Data); */
    /*   // errs() << Data << "\n"; */
    /*   // errs() << "\n"; */
    /*   AccessDetailFile >> Data; */
    /*   errs() << Data << "\n"; */
    /*   KernelParamUsageInKernel[KernelName].push_back( */
    /*       std::make_pair(ParamNumber, AccessInfo)); */
    /* } */

    /* std::ifstream ReuseDetailFile("reuse_detail_file.lst"); */
    /* errs() << "REUSE ANALYSIS FORM DEVICE\n"; */
    /* for (std::string Line; std::getline(ReuseDetailFile, Line);) { */
    /*   // errs() << Line << "\n"; */
    /*   std::stringstream LineStream(Line); */
    /*   std::string Token; */
    /*   std::getline(LineStream, Token, ' '); */
    /*   errs() << "KERNEL NAME: " << Token.erase(0, 4) << "\n"; */
    /*   std::string KernelName = Token; */
    /*   std::getline(LineStream, Token, ' '); */
    /*   errs() << "PARAM #: " << Token << "\n"; */
    /*   unsigned ParamNumber = stoi(Token); */
    /*   std::getline(LineStream, Token, ' '); */
    /*   errs() << "access ID: " << Token << "\n"; */
    /*   unsigned AccessID = std::stoi(Token); */
    /*   std::getline(LineStream, Token, ' '); */
    /*   errs() << "#access : " << Token << "\n"; */
    /*   unsigned NumberOfAccess = std::stoi(Token); */
    /*   std::getline(LineStream, Token, ' '); */
    /*   errs() << "INDEX : " << Token << "\n"; */
    /*   IndexAxisType ReuseIndexAxis = StringToIndexAxisType[Token]; */
    /*   for (std::string Token; std::getline(LineStream, Token, ' ');) { */
    /*     errs() << "Multiplier " << Token << "\n"; */
    /*     KernelParamReuseInKernel[KernelName][ParamNumber][ReuseIndexAxis] */
    /*       .push_back(Token); */
    /*   } */
    /*   errs() << "\n" */
    /*     << "\n"; */
    /* } */
  }

  void parseReuseDetailFile() {
    std::ifstream ResuseDetailFile("reuse_detail_file.lst");
    errs() << "REUSE ANALYSIS FORM DEVICE\n";
    for (std::string Line; std::getline(ResuseDetailFile, Line);) {
      // errs() << Line << "\n";
      std::stringstream LineStream(Line);
      std::string Token;
      std::getline(LineStream, Token, ' ');
      errs() << "KERNEL NAME: " << Token.erase(0, 4) << "\n";
      std::getline(LineStream, Token, ' ');
      errs() << "PARAM #: " << Token << "\n";
      std::getline(LineStream, Token, ' ');
      errs() << "INDEX : " << Token << "\n";
      for (std::string Token; std::getline(LineStream, Token, ' ');) {
        errs() << "Multiplier " << Token << "\n";
      }
      errs() << "\n"
             << "\n";
    }
  }

  void processKernelSignature(CallBase *I) {
    /* errs() << "CALL \n"; */
    /* I->dump(); */
    /* errs() << "SIGNATURE \n"; */
    auto *KernelPointer = I->getArgOperand(0);
    if (auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer)) {
      // KernelFunction->dump();
      KernelFunction->getFunctionType()->dump();
    }
  }

  void traverseGridSizeArgument(Value *GridSizeArgument) {
    errs() << "traverse grid size arg\n";
    GridSizeArgument->dump();
  }

  void parseGridSizeArgument(Value *GridSizeArgument, CallBase *CI) {
    errs() << "parsing grid size argrument\n";
    GridSizeArgument->dump();
    if (auto GridSizeOp = dyn_cast<Instruction>(GridSizeArgument)) {
      if (GridSizeOp->getOpcode() == Instruction::Mul) {
        errs() << "MUL\n";
        for (auto &Operand : GridSizeOp->operands()) {
          /* Operand->dump(); */
          if (auto ConstOper = dyn_cast<ConstantInt>(Operand)) {
            /* errs() << "is constant\n"; */
            /* errs() << ConstOper->getSExtValue() << "\n"; */
            if (ConstOper->getSExtValue() == 4294967297) {
              errs() << "magic duplication operation\n";
              for (auto &OtherOpCandidate : GridSizeOp->operands()) {
                if (OtherOpCandidate != ConstOper) {
                  traverseGridSizeArgument(OtherOpCandidate);
                  KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX] =
                      OtherOpCandidate;
                  KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY] =
                      OtherOpCandidate;
                }
              }
            }
          }
        }
      }
      if (GridSizeOp->getOpcode() == Instruction::Or) {
        errs() << "OR\n";
        for (auto &Operand : GridSizeOp->operands()) {
          /* Operand->dump(); */
          if (auto ConstOper = dyn_cast<ConstantInt>(Operand)) {
            /* errs() << "is constant\n"; */
            /* errs() << ConstOper->getSExtValue() << "\n"; */
            if (ConstOper->getSExtValue() ==
                4294967296) { // hard coded. ideally use arithemtic to figure
              errs() << "magic operation\n";
              for (auto &OtherOpCandidate : GridSizeOp->operands()) {
                if (OtherOpCandidate != ConstOper) {
                  errs() << "magic operation pushed\n";
                  CI->dump();
                  traverseGridSizeArgument(OtherOpCandidate);
                  KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX] =
                      OtherOpCandidate;
                  KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY] =
                      1; // hardcoded.
                }
              }
            }
          }
        }
      }
    }
  }

  void processKernelShapeArguments(Function &F) {
    errs() << "process kernel shape arguments\n";

    std::vector<CallBase *> PushCall;
    std::vector<CallBase *> PopCall;
    std::vector<CallBase *> LaunchCall;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallBase>(&I)) {
          auto *Callee = CI->getCalledFunction();
          if (Callee) {
            // errs() << Callee->getName() << "\n";
          }
          if (Callee && Callee->getName() == ("__cudaPushCallConfiguration")) {
            // errs() << Callee->getName() << "\n";
            PushCall.push_back(CI);
          }
          if (Callee && Callee->getName() == ("__cudaPopCallConfiguration")) {
            // errs() << Callee->getName() << "\n";
            PopCall.push_back(CI);
          }
          if (Callee && Callee->getName() == ("cudaLaunchKernel")) {
            // errs() << Callee->getName() << "\n";
            LaunchCall.push_back(CI);
          }
        }
      }
    }

    /* for (unsigned long Index = 0; Index < PushCall.size(); Index++) { */
    /*   errs() << "TRIPLE " << Index << "\n"; */
    /*   PushCall[Index]->dump(); */
    /*   PopCall[Index]->dump(); */
    /*   LaunchCall[Index]->dump(); */
    /* } */

    // Parsing the SROA. Very weird. No wonder no one wants to static analysis
    // on LLVM CUDA.PopCall
    for (unsigned long Index = 0; Index < PushCall.size(); Index++) {
      errs() << "TRIPLE " << Index << "\n";
      unsigned GridDimX, GridDimY, GridDimZ = 0;
      unsigned BlockDimX = 0, BlockDimY = 0, BlockDimZ = 0;
      Value *GridXYValue = PushCall[Index]->getOperand(0);
      PushCall[Index]->dump();
      GridXYValue->dump();
      KernelInvocationToGridDimXYValueMap[LaunchCall[Index]] = GridXYValue;
      if (auto *GridXYConst = dyn_cast<ConstantInt>(GridXYValue)) {
        unsigned long long GridXY = GridXYConst->getSExtValue();
        GridDimY = GridXY >> 32;
        GridDimX = (GridXY << 32) >> 32;
        errs() << "Grid X = " << GridDimX << "\n";
        errs() << "Grid Y = " << GridDimY << "\n";
        KernelInvocationToGridSizeMap[LaunchCall[Index]][AXIS_TYPE_GDIMX] =
            GridDimX;
        KernelInvocationToGridSizeMap[LaunchCall[Index]][AXIS_TYPE_GDIMY] =
            GridDimY;
      } else {
        errs() << "heh\n";
        parseGridSizeArgument(GridXYValue, LaunchCall[Index]);
      }
      Value *GridZValue = PushCall[Index]->getOperand(1);
      if (auto *GridZConst = dyn_cast<ConstantInt>(GridZValue)) {
        unsigned long GridZ = GridZConst->getSExtValue();
        GridDimZ = GridZ;
        errs() << "Grid Z = " << GridDimZ << "\n";
        KernelInvocationToGridSizeMap[LaunchCall[Index]][AXIS_TYPE_GDIMZ] =
            GridDimZ;
      } else {
        static_assert(true, "NO reach here. GRID DIM must be constant \n");
      }
      GridZValue->dump();
      KernelInvocationToGridDimZValueMap[LaunchCall[Index]] = GridZValue;
      Value *BlockXYValue = PushCall[Index]->getOperand(2);
      BlockXYValue->dump();
      if (auto *BlockXYConst = dyn_cast<ConstantInt>(BlockXYValue)) {
        unsigned long long BlockXY = BlockXYConst->getSExtValue();
        BlockDimY = BlockXY >> 32;
        BlockDimX = (BlockXY << 32) >> 32;
        errs() << "Block X = " << BlockDimX << "\n";
        errs() << "Block Y = " << BlockDimY << "\n";
      } else {
        static_assert(true, "NO reach here. BLOCK DIM must be constant \n");
      }
      Value *BlockZValue = PushCall[Index]->getOperand(3);
      BlockZValue->dump();
      if (auto *BlockZConst = dyn_cast<ConstantInt>(BlockZValue)) {
        unsigned long BlockZ = BlockZConst->getSExtValue();
        BlockDimZ = BlockZ;
        errs() << "Block Z = " << BlockDimZ << "\n";
      } else {
        static_assert(true, "NO reach here. GRID DIM must be constant \n");
      }
      KernelInvocationToBlockSizeMap[LaunchCall[Index]][AXIS_TYPE_BDIMX] =
          BlockDimX;
      KernelInvocationToBlockSizeMap[LaunchCall[Index]][AXIS_TYPE_BDIMY] =
          BlockDimY;
      KernelInvocationToBlockSizeMap[LaunchCall[Index]][AXIS_TYPE_BDIMZ] =
          BlockDimZ;
    }
  }

  void processKernelArguments(CallBase *I) {
    errs() << "Process kernel arguments\n";
    /* errs() << "CALL \n"; */
    I->dump();
    /* errs() << "NAME \n"; */
    auto *KernelPointer = I->getArgOperand(0);
    if (auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer)) {
      auto KernelName = KernelFunction->getName();
      /* errs() << getOriginalKernelName(KernelName.str()) << "\n"; */
    }
    /* errs() << "ARG STRUCT \n"; */
    auto *KernelArgs =
        I->getArgOperand(5); // the 5th argument is the kernel argument struct.
    errs() << "selected kernel argument\n";
    KernelArgs->dump();
    KernelInvocationToStructMap[I] = KernelArgs;
    /* errs() << "USERS \n"; */
    /* for (llvm::User *Karg : KernelArgs->users()) { */
    /*   errs() << "user: "; */
    /*   Karg->dump(); */
    /*   recurseTillStoreOrEmtpy(I, KernelArgs, Karg, Karg); */
    /* } */
    findAllocationOnLocalStack(I, KernelArgs);
    // for (auto &Arg: I->args()){
    //   Arg->dump();
    // }
    return;
  }

  unsigned long int getAllocationSize(Value *PointerOp) {
    errs() << "get alloation size (pointer) \n";
    PointerOp->dump();
    auto *OriginalPointer = PointerOpToOriginalPointers[PointerOp];
    OriginalPointer->dump();
    if (StructAllocas.find(OriginalPointer) != StructAllocas.end()) {
      if (PointerOpToOriginalStructPointersIndex.find(PointerOp) !=
          PointerOpToOriginalStructPointersIndex.end()) {
        auto argnum = PointerOpToOriginalStructPointersIndex[PointerOp];
        errs() << "faund: " << argnum << "\n";
        unsigned long int AllocationSize =
            MallocPointerStructToIndexToSizeMap[OriginalPointer][argnum];
        return AllocationSize;
      }
    }
    unsigned long int AllocationSize = MallocPointerToSizeMap[OriginalPointer];
    return AllocationSize;
  }

  unsigned long int getAllocationSize(CallBase *CI, unsigned argid) {
    errs() << "get alloation size\n";
    auto ArgNumberToAllocationMap =
        KernelInvocationToArgNumberToAllocationMap[CI];
    auto PointerOp = ArgNumberToAllocationMap[argid];
    PointerOp->dump();
    auto *OriginalPointer = PointerOpToOriginalPointers[PointerOp];
    OriginalPointer->dump();
    if (StructAllocas.find(OriginalPointer) != StructAllocas.end()) {
      errs() << "faund: " << PointerOpToOriginalStructPointersIndex[PointerOp]
             << "\n";
    }
    unsigned long int AllocationSize = MallocPointerToSizeMap[OriginalPointer];
    return AllocationSize;
  }

  long long operate(BinaryOperator *BO, int v1, int v2) {
    long long res;
    /* errs() << v1 << "  " << v2 << "\n"; */
    if (BO->getOpcode() == Instruction::Mul) {
      res = v1 * v2;
    }
    if (BO->getOpcode() == Instruction::SDiv) {
      res = v2 / v1;
    }
    if (BO->getOpcode() == Instruction::UDiv) {
      res = v2 / v1;
    }
    if (BO->getOpcode() == Instruction::Sub) {
      res = v2 - v1;
    }
    if (BO->getOpcode() == Instruction::Add) {
      res = v1 + v2;
    }
    if (BO->getOpcode() == Instruction::LShr) {
      res = v2 >> v1;
    }
    /* errs() << "resu = " << res << "\n"; */
    return res;
  }

  // iter 0 has phi node with value initial
  long long evaluateRPNForIter0(CallBase *CI, std::vector<Value *> RPN) {
    /* std::vector<Value*> RPN = RRPN; */
    reverse(RPN.begin(), RPN.end());
    bool phiseen = false;

    std::stack<long long> stack;

    for (auto Token = RPN.begin(); Token != RPN.end(); Token++) {
      // if terminal (host side terminals only include function arguments)
      (*Token)->dump();
      if (TerminalValues.find(*Token) != TerminalValues.end()) {
        auto ActualArg = FormalArgumentToActualArgumentMap[*Token][0];
        ActualArg->dump();
        if (ConstantInt *CoI = dyn_cast<ConstantInt>(ActualArg)) {
          /* errs() << "Yayaya " << CoI->getSExtValue() << "\n"; */
          stack.push(CoI->getSExtValue());
          /* errs() << "stack push = " << CoI->getSExtValue() << "\n"; */
        } else {
          /* errs() << "PANIC!!!!\n"; */
          errs() << "NOt a constant, so checking for values\n";
          if (PointerOpToOriginalConstant.find(ActualArg) !=
              PointerOpToOriginalConstant.end()) {
            errs() << PointerOpToOriginalConstant[ActualArg] << "\n";
            stack.push(PointerOpToOriginalConstant[ActualArg]);
          }
        }
        continue;
      }
      if (ConstantInt *CoI = dyn_cast<ConstantInt>(*Token)) {
        stack.push(CoI->getSExtValue());
        /* errs() << "stack push = " << CoI->getSExtValue() << "\n"; */
        continue;
      }
      if (Instruction *I = dyn_cast<Instruction>(*Token)) {
        // order matters, probably
        if (PHINode *Phi = dyn_cast<PHINode>(I)) {
          if (phiseen == false) {
            stack.push(0); //
            /* errs() << "stack push = 0\n"; */
            phiseen = true;
            continue;
          }
          if (phiseen == true) {
            long long op1 = stack.top();
            /* errs() << "stack pop = " << op1 << "\n"; */
            stack.pop();
            long long op2 = stack.top();
            /* errs() << "stack pop = " << op2 << "\n"; */
            stack.pop();
            long long result = (op1 < op2) ? op1 : op2;
            stack.push(result);
            /* errs() << "stack push = " << result << "\n"; */
            continue;
          }
        }
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
          long long op1 = stack.top();
          /* errs() << "stack pop = " << op1 << "\n"; */
          stack.pop();
          long long op2 = stack.top();
          /* errs() << "stack pop = " << op2 << "\n"; */
          stack.pop();
          long long result = operate(BO, op1, op2);
          stack.push(result);
          /* errs() << "stack push = " << result << "\n"; */
          continue;
        }
      }
    }
    /* return 0; */
    return stack.top();
  }

  std::vector<Value *> getExpressionTree(Value *V) {
    std::vector<Value *> RPN(0);
    std::stack<Value *> Stack;
    std::set<Value *> Visited;
    std::set<Value *> PhiNodesVisited;

    errs() << "Getting Expression Tree\n";
    Stack.push(V);

    while (!Stack.empty()) {
      Value *Current = Stack.top();
      Current->dump();
      Stack.pop();
      if (PhiNodesVisited.find(Current) != PhiNodesVisited.end()) {
        RPN.push_back(Current);
        continue;
      }
      if (Visited.find(Current) != Visited.end()) {
        errs() << "hi\n";
        continue;
      }
      RPN.push_back(Current);
      if (TerminalValues.find(Current) != TerminalValues.end()) {
        continue;
      }
      // iterate through operands
      if (isa<Instruction>(Current)) {
        auto *In = dyn_cast<Instruction>(Current);
        if (auto *LI = dyn_cast<LoadInst>(In)) {
          Stack.push(LI->getPointerOperand());
        } else if (auto *SI = dyn_cast<StoreInst>(In)) {
          Stack.push(SI->getPointerOperand());
        } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(In)) {
          for (int i = 1; i < GEPI->getNumIndices() + 1;
               i++) { // indices not includes the pointer
            Stack.push(GEPI->getOperand(i));
          }
        } else if (auto *Phi = dyn_cast<PHINode>(In)) {
          for (auto &Operand : In->operands()) {
            Stack.push(Operand);
          }
          PhiNodesVisited.insert(Phi);
        } else {
          for (auto &Operand : In->operands()) {
            Stack.push(Operand);
          }
        }
        Visited.insert(Current);
        continue;
      }
    }

    errs() << "RPN \n";
    for (auto RPNIter = RPN.begin(); RPNIter != RPN.end(); RPNIter++) {
      if (TerminalValues.find(*RPNIter) != TerminalValues.end() ||
          isa<ConstantInt>(*RPNIter)) {
        errs() << "terminal ";
      } else {
        errs() << "operand ";
      }
      (*RPNIter)->dump();
    }
    errs() << "\n";

    return RPN;
  }

  // be careful when you use this with malloc, double pointer and all
  void insertCodeToPrintAddress(CallBase *CI, Value *P) {
    // TODO: add asserts
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    auto V = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    auto printIntFunc = F->getParent()->getOrInsertFunction(
        "print_value_i64", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
    Value *Args[] = {V};
    auto PrintFunc = Builder.CreateCall(printIntFunc, Args);
    return;
  }

  void insertCodeToPrintSize(CallBase *CI, Value *V) {
    // TODO: add asserts
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::FunctionCallee PrintIntFunc = F->getParent()->getOrInsertFunction(
        "print_value_i64", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
    Value *Args[] = {V};
    llvm::CallInst *PrintFunc = Builder.CreateCall(PrintIntFunc, Args);
    return;
  }

  void insertCodeToPrintGenericInt32(Instruction *CI, Value *V) {
    assert(V->getType()->isIntegerTy(32));
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    llvm::FunctionCallee PrintIntFunc = F->getParent()->getOrInsertFunction(
        "print_value_i32", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));
    Value *Args[] = {V};
    llvm::CallInst *PrintFunc = Builder.CreateCall(PrintIntFunc, Args);
    return;
  }

  void insertCodeToPrintGenericInt64(Instruction *CI, Value *V) {
    assert(V->getType()->isIntegerTy(64));
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    llvm::FunctionCallee PrintIntFunc = F->getParent()->getOrInsertFunction(
        "print_value_i64", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
    Value *Args[] = {V};
    llvm::CallInst *PrintFunc = Builder.CreateCall(PrintIntFunc, Args);
    return;
  }

  void insertCodeToPrintGenericFloat32(Instruction *CI, Value *V) {
    assert(V->getType()->isFloatTy());
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    llvm::FunctionCallee PrintF32Func = F->getParent()->getOrInsertFunction(
        "print_value_f32", Type::getVoidTy(Ctx), Type::getFloatTy(Ctx));
    Value *Args[] = {V};
    llvm::CallInst *PrintFunc = Builder.CreateCall(PrintF32Func, Args);
    return;
  }

  void insertCodeToPrintGenericFloat64(Instruction *CI, Value *V) {
    assert(V->getType()->isDoubleTy());
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    llvm::FunctionCallee PrintF64Func = F->getParent()->getOrInsertFunction(
        "print_value_f64", Type::getVoidTy(Ctx), Type::getDoubleTy(Ctx));
    Value *Args[] = {V};
    llvm::CallInst *PrintFunc = Builder.CreateCall(PrintF64Func, Args);
    return;
  }

  void addCodeToAddInvocationID(CallBase *CI, unsigned InvocationID) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    auto V = Builder.getInt32(InvocationID);
    auto printIntFunc = F->getParent()->getOrInsertFunction(
        "add_invocation_id", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx));
    Value *Args[] = {V};
    auto PrintFunc = Builder.CreateCall(printIntFunc, Args);
    return;
  }

  // TODO: Fix this mess ASAP
  // First get the pointer to the allocation, not the pointer to the pointer!
  void insertCodeToRecordMalloc(CallBase *CI, Value *P, Value *S) {
    // TODO: add asserts
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    if(CI->getNextNode()) { // TODO: check for invoke inst (instead of call) , nto this ugly hack.
        IRBuilder<> Builder(CI);
        Builder.SetInsertPoint(CI->getNextNode());
        P->getType()->dump();
        llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
        Value *Args[] = {Ptr, S};
        // Builder.CreateCall(Fn, Args);
        llvm::FunctionCallee AddIntoAllocFunc = F->getParent()->getOrInsertFunction(
                "addIntoAllocationMap", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx),
                Type::getInt64Ty(Ctx));
        Builder.CreateCall(AddIntoAllocFunc, Args);
        llvm::FunctionCallee PrintAllocFunc = F->getParent()->getOrInsertFunction(
                "printAllocationMap", Type::getVoidTy(Ctx));
        ArrayRef<Value *> PrintArgs = {};
        Builder.CreateCall(PrintAllocFunc, PrintArgs);
    } else { // else it is an invoke inst
        // get the output of malloc, which is in P, find the successors of this BB,
        auto BB = CI->getParent();
        errs() << "BB\n";
        BB->dump();
        errs() << "BB over\n";
        InvokeInst* II = dyn_cast<InvokeInst>(CI);
        BasicBlock* succ = II->getNormalDest();
        errs() << "BB succ\n";
        succ->dump();
        errs() << "BB over\n";
        auto IP = succ->getFirstNonPHI();
        IP->dump();
        errs() << "BB fi over\n";
        IRBuilder<> Builder(IP);
        P->getType()->dump();
        llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
        Value *Args[] = {Ptr, S};
        // Builder.CreateCall(Fn, Args);
        llvm::FunctionCallee AddIntoAllocFunc = F->getParent()->getOrInsertFunction(
                "addIntoAllocationMap", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx),
                Type::getInt64Ty(Ctx));
        Builder.CreateCall(AddIntoAllocFunc, Args);
        llvm::FunctionCallee PrintAllocFunc = F->getParent()->getOrInsertFunction(
                "printAllocationMap", Type::getVoidTy(Ctx));
        ArrayRef<Value *> PrintArgs = {};
        Builder.CreateCall(PrintAllocFunc, PrintArgs);
    }
    return;
  }

  void insertCodeToAddAccessCount(Instruction *Location, unsigned aid, Value *P, Value *S) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr, S};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddACToAllocFunc = F->getParent()->getOrInsertFunction(
        "addACToAllocation", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx),
        Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddACToAllocFunc, Args);
    auto AID = Builder.getInt32(aid);
    Value *Args2[] = {AID, Ptr};
    llvm::FunctionCallee AddAIDToAllocation = F->getParent()->getOrInsertFunction(
        "add_aid_allocation_map", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddAIDToAllocation, Args2);
    Value *Args3[] = {AID, S};
    llvm::FunctionCallee AddAIDToAC = F->getParent()->getOrInsertFunction(
        "add_aid_ac_map", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddAIDToAC, Args3);
    // llvm::FunctionCallee PrintAllocFunc = F->getParent()->getOrInsertFunction(
        // "printACToAllocationMap", Type::getVoidTy(Ctx));
    // ArrayRef<Value *> PrintArgs = {};
    // Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToAddAccessCountPerAccess(CallBase *CI, unsigned aid,
                                           Value *ac) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::Value *AID = Builder.getInt32(aid);
    Value *Args[] = {AID, ac};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddACToAID = F->getParent()->getOrInsertFunction(
        "add_aid_ac_map", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt32Ty(Ctx));
    Builder.CreateCall(AddACToAID, Args);
    return;
  }

  void insertCodeToAddPDBIDX(Instruction *Location, Value *P, Value *S) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr, S};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddACToAllocFunc = F->getParent()->getOrInsertFunction(
        "add_pd_bidx_to_allocation", Type::getVoidTy(Ctx),
        Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddACToAllocFunc, Args);
    // llvm::FunctionCallee PrintAllocFunc =
    // F->getParent()->getOrInsertFunction(
    //     "print_pd_bidx_map", Type::getVoidTy(Ctx));
    // ArrayRef<Value *> PrintArgs = {};
    // Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToAddPDBIDY(Instruction *CI, Value *P, Value *S) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr, S};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddACToAllocFunc = F->getParent()->getOrInsertFunction(
        "add_pd_bidy_to_allocation", Type::getVoidTy(Ctx),
        Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddACToAllocFunc, Args);
    // llvm::FunctionCallee PrintAllocFunc =
    // F->getParent()->getOrInsertFunction(
    //     "print_pd_bidy_map", Type::getVoidTy(Ctx));
    // ArrayRef<Value *> PrintArgs = {};
    // Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToAddPDPhi(Instruction *CI, Value *P, Value *S) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr, S};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddACToAllocFunc = F->getParent()->getOrInsertFunction(
        "add_pd_phi_to_allocation", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx),
        Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddACToAllocFunc, Args);
    // llvm::FunctionCallee PrintAllocFunc =
    // F->getParent()->getOrInsertFunction(
    //     "print_pd_phi_map", Type::getVoidTy(Ctx));
    // ArrayRef<Value *> PrintArgs = {};
    // Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToAddWSS(Instruction *Location, Value *P, Value *S, Value* A) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr, S, A};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddWSS = F->getParent()->getOrInsertFunction(
        "add_wss_to_map", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx),
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx));
    Builder.CreateCall(AddWSS, Args);
    llvm::FunctionCallee PrintAllocFunc = F->getParent()->getOrInsertFunction(
        "print_wss_map", Type::getVoidTy(Ctx));
    ArrayRef<Value *> PrintArgs = {};
    Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToSetPChase(Instruction *Location, unsigned AID , Value* P, bool pChase) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    Value* aidValue = insertConstantNode(Location, AID);
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value* pChaseValue = insertConstantNode(Location, pChase);
    Value *Args[] = {aidValue, Ptr, pChaseValue};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddPC = F->getParent()->getOrInsertFunction(
        "add_aid_pchase_map", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt64Ty(Ctx), Type::getInt1Ty(Ctx));
    Builder.CreateCall(AddPC, Args);
    return;
  }

  void insertCodeToSetIncomp(Instruction *Location, unsigned AID , bool incomp) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    Value* aidValue = insertConstantNode(Location, AID);
    Value* pChaseValue = insertConstantNode(Location, incomp);
    Value *Args[] = {aidValue, pChaseValue};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee AddPC = F->getParent()->getOrInsertFunction(
        "add_aid_ac_incomp_map", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt1Ty(Ctx));
    Builder.CreateCall(AddPC, Args);
    return;
  }

  void insertCodeToAddWSS_iterdep(Instruction* Location, unsigned aid, Value* wss) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    // Builder.CreateCall(Fn, Args);
    llvm::Value *AID = Builder.getInt32(aid);
    Value *Args[] = {AID, wss};
    llvm::FunctionCallee AddWSS = F->getParent()->getOrInsertFunction(
        "add_aid_wss_map_iterdep", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt32Ty(Ctx));
    Builder.CreateCall(AddWSS, Args);
    // llvm::FunctionCallee PrintAllocFunc = F->getParent()->getOrInsertFunction(
    //     "print_aid_wss_map_iterdep", Type::getVoidTy(Ctx));
    // ArrayRef<Value *> PrintArgs = {};
    // Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToAddAIDToInvocationID(Instruction *Location, unsigned aid,
                                        unsigned invid) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    // Builder.CreateCall(Fn, Args);
    llvm::Value *AID = Builder.getInt32(aid);
    llvm::Value *INVID = Builder.getInt32(invid);
    Value *Args[] = {AID, INVID};
    llvm::FunctionCallee AddAIDToInvocationID = F->getParent()->getOrInsertFunction(
        "add_aid_invocation_map", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt32Ty(Ctx));
    Builder.CreateCall(AddAIDToInvocationID, Args);
    // llvm::FunctionCallee PrintAllocFunc =
    // F->getParent()->getOrInsertFunction(
    //     "print_aid_wss_map_iterdep", Type::getVoidTy(Ctx));
    // ArrayRef<Value *> PrintArgs = {};
    // Builder.CreateCall(PrintAllocFunc, PrintArgs);
    return;
  }

  void insertCodeToProcessWSS_iterdep(Instruction* Location) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee ProcessWSS_iter = F->getParent()->getOrInsertFunction(
        "process_iterdep_access", Type::getVoidTy(Ctx));
    ArrayRef<Value *> Args = {};
    Builder.CreateCall(ProcessWSS_iter, Args);
    return;
  }

  // This function should get all the information it needs from the runtime, not
  // from LLVM values
  // must be called once per iteration
  void insertCodeToPerformInvocationMemoryMgmt(Instruction *Location, CallBase  *CI) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::ConstantInt *MemSize = Builder.getInt64(6 * 1024ULL * 1024ULL * 1024ULL);
    unsigned invid = KernelInvocationToInvocationIDMap[CI];
    auto InvID = Builder.getInt32(invid);
    ArrayRef<Value *> Args = {MemSize, InvID};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee MemMgmtFn = F->getParent()->getOrInsertFunction(
        "perform_memory_management", Type::getVoidTy(Ctx),
        Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx));
    Builder.CreateCall(MemMgmtFn, Args);
    return;
  }

  // call this in the Loop, with for each memory allocation as the operatn
  Instruction *insertCodeToPerformMemoryMgmtIteration(Instruction *Location,
                                                      Value *Iter) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    ArrayRef<Value *> Args = {Iter};
    llvm::FunctionCallee MemMgmtFn = F->getParent()->getOrInsertFunction(
        "penguinSuperPrefetchWrapper", Type::getVoidTy(Ctx),
        Type::getInt32Ty(Ctx));
    return Builder.CreateCall(MemMgmtFn, Args);
  }

  // This function should get all the information it needs from the runtime, not
  // from LLVM values
  // must be called once per iteration
  Instruction* insertCodeToPerformGlobalMemoryMgmt(Instruction *Location) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::ConstantInt *MemSize =
        Builder.getInt64(4 * 1024ULL * 1024ULL * 1024ULL);
    ArrayRef<Value *> Args = {MemSize};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee MemMgmtFn = F->getParent()->getOrInsertFunction(
        "perform_memory_management_global", Type::getVoidTy(Ctx),
        Type::getInt64Ty(Ctx));
    return Builder.CreateCall(MemMgmtFn, Args);
  }

  Instruction* insertCodeToPerformIterativeMemoryMgmt(Instruction *Location) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::ConstantInt *MemSize =
        Builder.getInt64(6 * 1024ULL * 1024ULL * 1024ULL);
    ArrayRef<Value *> Args = {MemSize};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee MemMgmtFn = F->getParent()->getOrInsertFunction(
        "perform_memory_management_iterative", Type::getVoidTy(Ctx),
        Type::getInt64Ty(Ctx));
    return Builder.CreateCall(MemMgmtFn, Args);
  }

  // the arguments to penguin super prefetch must come from runtime maps
  void insertCodeToPenguinSuperPrefetch(Instruction *Location) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    return;
  }

  Value *insertTreeEvaluationCodeUsingCoeffecientVectors(
      CallBase *CI, std::map<ExprTreeNode *, Value *> Unknowns,
      ExprTreeNode *Node) {
    // for each of ETO_BIDX, ETO_BIDY, ETO_BIDZ, ETO_TIDX, ETO_TIDY, ETO_TIDZ,
    // identify the co-efficients by traversing up the tree
    return nullptr;
  }

  Value *insertTreeEvaluationCode(Instruction *Location, CallBase *CI,
                                  std::map<ExprTreeNode *, Value *> Unknowns,
                                  ExprTreeNode *node,
                                  Value *LoopIters = nullptr) {
    if (node == nullptr)
      return nullptr;
    errs() << "handling node " << node->original_str << "\n";
    if (isTerminal(node)) {
      // handle this node
      errs() << "iliec: " << node->original_str << "\n";
      if (Unknowns.find(node) != Unknowns.end()) {
        Value *val = Unknowns[node];
        val->dump();
        return val;
      }
      if (node->op == ETO_CONST) { // currently assuming all constants are 32
                                   // bit unsigned integers
        errs() << "node value = " << node->value << "\n";
        node->value =
            stoll(node->original_str); // TODO: remove this hack: ensure that
                                      // the value is set correctly at some
                                      // earlier point
        return insertConstantNode(Location, node);
      }
      // getting max
      if (node->op == ETO_PHI_TERM) {
        errs() << "PHI TERM\n";
        return insertConstantNode(Location, unsigned (0));
      }
      assert(false); // must not reach here.
    } else {
      if (node->op == ETO_PHI) {
        errs() << "PHI (hello hello)\n";
        // TODO: get the phi node to loop id to loop iterations
        // Each phi node (numbered)
        /* // LoopIters->dump(); */
        auto one = insertConstantNode(Location, (unsigned)(1));
        /* return insertComputationNode(Location, LoopIters, one, ETO_SUB); */
        return one;
        // return LoopIters;
      }
      errs() << "childrens:";
      errs() << node->children[0]->original_str << " "
             << node->children[1]->original_str << "\n";
      llvm::Value *Left = insertTreeEvaluationCode(
          Location, CI, Unknowns, node->children[0], LoopIters);
      llvm::Value *Right = insertTreeEvaluationCode(
          Location, CI, Unknowns, node->children[1], LoopIters);
      // handle this node
      // create a new LLVM value (Instruction) with the operation in question
      // it will take the left and right children as operands
      // return the new LLVM value
      auto cn = insertComputationNode(Location, Left, Right, node->op);
      return cn;
    }
    return nullptr;
  }

  Instruction *insertMaximumOfTwo(Instruction *Location,
          Value *A, Value* B) {
      Function *F = Location->getParent()->getParent();
      LLVMContext &Ctx = F->getContext();
      IRBuilder<> Builder(Location);
      ArrayRef<Value *> Args = {A, B};
      llvm::FunctionCallee MaxOfTwo = F->getParent()->getOrInsertFunction(
              "larger_of_two", Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
              Type::getInt32Ty(Ctx));
      return Builder.CreateCall(MaxOfTwo, Args);
  }

  Instruction *insertMinimumOfTwo(Instruction *Location,
          Value *A, Value* B) {
      Function *F = Location->getParent()->getParent();
      LLVMContext &Ctx = F->getContext();
      IRBuilder<> Builder(Location);
      ArrayRef<Value *> Args = {A, B};
      llvm::FunctionCallee MinOfTwo = F->getParent()->getOrInsertFunction(
              "smaller_of_two", Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
              Type::getInt32Ty(Ctx));
      return Builder.CreateCall(MinOfTwo, Args);
  }

  Value* computeSubExpression(Instruction* Location, CallBase* CI,
          std::map<ExprTreeNodeAdvanced*, Value*> Unknowns,
          ExprTreeNodeAdvanced* node) {
      errs() << "computeSubExpression\n";
      if(isTerminal(node)) {
          errs() << "iliec: " << node->original_str << "\n";
          if (Unknowns.find(node) != Unknowns.end()) {
              errs() << "found unknown\n";
              Value *val = Unknowns[node];
              val->dump();
              return val;
          }
          if (node->op == ETO_CONST) { // currently assuming all constants are 32
                                       // bit unsigned integers
              errs() << "node value = " << node->value << "\n";
              node->value =
                  stoll(node->original_str); // TODO: remove this hack: ensure that
                                             // the value is set correctly at some
                                             // earlier point
              return insertConstantNode(Location, node);
          }
          assert(false);
      } else if(isOperation(node)) {
          /* errs() << node->children[0]->original_str << " " */
          /* << node->children[1]->original_str << "\n"; */
          llvm::Value *Left = computeSubExpression(
                  Location, CI, Unknowns, node->children[0]);
          llvm::Value *Right = computeSubExpression(
                  Location, CI, Unknowns, node->children[1]);
          // handle this node
          // create a new LLVM value (Instruction) with the operation in question
          // it will take the left and right children as operands
          // return the new LLVM value
          auto cn = insertComputationNodeAdvanced(Location, Left, Right, node->op);
          return cn;
      } else { // single child operations
          llvm::Value *Child = computeSubExpression(
                  Location, CI, Unknowns, node->children[0]);
          return Child;
      }
      return nullptr;
  }

  ///// UNknows will be minimum
  Value* computeSmallestValueForTerminalPhi(Instruction* Location,
          CallBase* CI, std::map<ExprTreeNodeAdvanced*, Value*> Unknowns,
          ExprTreeNodeAdvanced* node, std::map<unsigned, Value*> LoopIDToNumIterationsMap, Value* totalIncrementOfPhi) {
      assert(node != nullptr);
      ExprTreeNodeAdvanced* parent = node->parent;
      ExprTreeNodeAdvanced* current = node;
      Value* Accum = insertConstantNode(Location, unsigned (0));
      while(parent->op != ETO_PHI) {
          current = parent;
          parent = current->parent;
      }
      ExprTreeNodeAdvanced* otherChildOfParentPhi;
      if(parent->children[0] == current) {
          otherChildOfParentPhi = parent->children[1];
      } else {
          otherChildOfParentPhi = parent->children[0];
      }
      auto otherChildMin = insertTreeEvaluationCodeAdvanced(Location, CI, Unknowns, otherChildOfParentPhi, true, LoopIDToNumIterationsMap);  // true for minimize. change to enum
      Accum = insertComputationNode(Location, totalIncrementOfPhi, otherChildMin, ETO_ADD);
      return Accum;
  }

  Value* computeLargestValueForTerminalPhi(Instruction* Location,
          CallBase* CI, std::map<ExprTreeNodeAdvanced*, Value*> Unknowns,
          ExprTreeNodeAdvanced* node, std::map<unsigned, Value*> LoopIDToNumIterationsMap, Value* totalIncrementOfPhi) {
      assert(node != nullptr);
      ExprTreeNodeAdvanced* parent = node->parent;
      ExprTreeNodeAdvanced* current = node;
      Value* Accum = insertConstantNode(Location, unsigned (0));
      while(parent->op != ETO_PHI) {
          current = parent;
          parent = current->parent;
      }
      ExprTreeNodeAdvanced* otherChildOfParentPhi;
      if(parent->children[0] == current) {
          otherChildOfParentPhi = parent->children[1];
      } else {
          otherChildOfParentPhi = parent->children[0];
      }
      auto otherChildMax = insertTreeEvaluationCodeAdvanced(Location, CI, Unknowns, otherChildOfParentPhi, false, LoopIDToNumIterationsMap);  // true for minimize. change to enum
      Accum = insertComputationNode(Location, totalIncrementOfPhi, otherChildMax, ETO_ADD);
      return Accum;
  }

  Value* computePerIterationIncrementForTerminalPhi(Instruction* Location,
          CallBase* CI, std::map<ExprTreeNodeAdvanced*, Value*> Unknowns,
          ExprTreeNodeAdvanced* node, std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
      assert(node != nullptr);
      ExprTreeNodeAdvanced* parent = node->parent;
      ExprTreeNodeAdvanced* current = node;
      Value* Accum = insertConstantNode(Location, unsigned (0));
      while(parent->op != ETO_PHI) {
          errs() << "comutingn per iter incr\n";
          errs() << parent->original_str << "\n";
          // for the parent children, identify the child on other path
          ExprTreeNodeAdvanced* otherChild = nullptr;
          assert(parent->op == ETO_ADD); // Also 
          for(auto child = parent->children.begin(); child != parent->children.end(); child++) {
              if((*child) != current) {
                  otherChild = *child;
              }
          }
          Value* otherChildValue = computeSubExpression(Location, CI, Unknowns, otherChild);
          Accum = insertComputationNodeAdvanced(Location, Accum, otherChildValue, ETO_ADD);
          current = parent;
          parent = current->parent;
      }
      // TODO get loop count for the PHI node, and multiply with accum
      unsigned phiID = node->arg;
      unsigned loopID = PhiNodeToLoopIDMap[phiID];
      errs() << "per iteration increment, loop phi arg = " << phiID << "  " << loopID << "\n";
      insertCodeToPrintGenericInt32(Location, LoopIDToNumIterationsMap[loopID]);
      /* insertCodeToPrintGenericInt32(Location, Accum); */
      Accum = insertComputationNodeAdvanced(Location, Accum, LoopIDToNumIterationsMap[loopID], ETO_MUL);
      return Accum;
  }

  // this code assumes that PHI nodes are not dependent on other phi nodes
  Value* insertTreeEvaluationCodeForPhi(Instruction* Location, CallBase* CI,
          std::map<ExprTreeNodeAdvanced*, Value*> Unknowns,
          ExprTreeNodeAdvanced* node, bool rootphi, bool minimize, 
          std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
      if(node == nullptr) {
          return nullptr;
      }
      if(isTerminal(node)) {
          errs() << "iliec: " << node->original_str << "\n";
          if (Unknowns.find(node) != Unknowns.end()) {
              errs() << "found unknown\n";
              Value *val = Unknowns[node];
              val->dump();
              return val;
          }
          if (node->op == ETO_CONST) { // currently assuming all constants are 32
                                       // bit unsigned integers
              errs() << "node value = " << node->value << "\n";
              node->value =
                  stoll(node->original_str); // TODO: remove this hack: ensure that
                                             // the value is set correctly at some
                                             // earlier point
              return insertConstantNode(Location, node);
          }
          assert(false);
      }
      if(node->op == ETO_PHI && rootphi == false) {
          //type unsigned
          auto totalIncrementOfPhi = computePerIterationIncrementForTerminalPhi(Location, CI, Unknowns, node, LoopIDToNumIterationsMap);
          if(minimize) {
              // TODO: get the true value for root phi
              /* return insertConstantNode(Location, unsigned(0)); */
              return computeSmallestValueForTerminalPhi(Location, CI, Unknowns, node, LoopIDToNumIterationsMap, totalIncrementOfPhi);
          } else {
              return computeLargestValueForTerminalPhi(Location, CI, Unknowns, node, LoopIDToNumIterationsMap, totalIncrementOfPhi);
          }
          /* return insertConstantNode(Location, unsigned (0)); */
      }
      // not terminal
      errs() << "childrens:";

      for(auto child = node->children.begin(); child != node->children.end(); child++) {
          errs() << (*child)->original_str << " ";
      }
      if(node->op == ETO_PHI && rootphi == true) {
          errs() << "root phi\n";
          llvm::Value *Left = insertTreeEvaluationCodeForPhi(
                  Location, CI, Unknowns, node->children[0], false, minimize, LoopIDToNumIterationsMap);
          llvm::Value *Right = insertTreeEvaluationCodeForPhi(
                  Location, CI, Unknowns, node->children[1], false, minimize, LoopIDToNumIterationsMap);
          if(minimize) {
              auto cn = insertMinimumOfTwo(Location, Left, Right);
              return cn;
          } else {
              auto cn = insertMaximumOfTwo(Location, Left, Right);
              return cn;
          }
      } else if(isOperation(node)) {
          /* errs() << node->children[0]->original_str << " " */
          /* << node->children[1]->original_str << "\n"; */
          llvm::Value *Left = insertTreeEvaluationCodeForPhi(
                  Location, CI, Unknowns, node->children[0], false, minimize, LoopIDToNumIterationsMap);
          llvm::Value *Right = insertTreeEvaluationCodeForPhi(
                  Location, CI, Unknowns, node->children[1], false, minimize, LoopIDToNumIterationsMap);
          // handle this node
          // create a new LLVM value (Instruction) with the operation in question
          // it will take the left and right children as operands
          // return the new LLVM value
          auto cn = insertComputationNodeAdvanced(Location, Left, Right, node->op);
          return cn;
      } else { // single child operations
          llvm::Value *Child = insertTreeEvaluationCodeForPhi(
                  Location, CI, Unknowns, node->children[0], false, minimize, LoopIDToNumIterationsMap);
          return Child;
      }
      assert(false);
  }

  // function works for evaluating both max and min,
  // Unknowns contains max or min, depeneding on the need
  Value *insertTreeEvaluationCodeAdvanced(Instruction *Location, CallBase *CI,
                                  std::map<ExprTreeNodeAdvanced *, Value *> Unknowns,
                                  ExprTreeNodeAdvanced *node,
                                  bool minimize, std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
    if (node == nullptr)
      return nullptr;
    errs() << "handling node " << node->original_str << "\n";
    if (isTerminal(node)) {
      // handle this node
      errs() << "iliec: " << node->original_str << "\n";
      if (Unknowns.find(node) != Unknowns.end()) {
          errs() << "found unknown\n";
        Value *val = Unknowns[node];
        val->dump();
        return val;
      }
      if (node->op == ETO_CONST) { // currently assuming all constants are 32
                                   // bit unsigned integers
        errs() << "node value = " << node->value << "\n";
        node->value =
            stoll(node->original_str); // TODO: remove this hack: ensure that
                                      // the value is set correctly at some
                                      // earlier point
        return insertConstantNode(Location, node);
      }
      // getting max
      if (node->op == ETO_PHI_TERM) {
        errs() << "PHI TERM\n";
        return insertConstantNode(Location, unsigned (0));
      }
      assert(false); // must not reach here.
    } else {
      if (node->op == ETO_PHI) {
        errs() << "PHI (hello hello)\n";
        // TODO: get the phi node to loop id to loop iterations
        // Each phi node (numbered)
        /* // LoopIters->dump(); */
        /* auto one = insertConstantNode(Location, (unsigned)(1)); */
        /* return insertComputationNode(Location, LoopIters, one, ETO_SUB); */
        auto one = insertTreeEvaluationCodeForPhi(Location, CI, Unknowns, node, true, minimize, LoopIDToNumIterationsMap);
        return one;
        // return LoopIters;
      }
      errs() << "childrens:";

      for(auto child = node->children.begin(); child != node->children.end(); child++) {
          errs() << (*child)->original_str << " ";
      }
      errs() << "\n";
      if(isOperation(node)) {
          /* errs() << node->children[0]->original_str << " " */
          /* << node->children[1]->original_str << "\n"; */
          llvm::Value *Left = insertTreeEvaluationCodeAdvanced(
                  Location, CI, Unknowns, node->children[0], minimize, LoopIDToNumIterationsMap);
          llvm::Value *Right = insertTreeEvaluationCodeAdvanced(
                  Location, CI, Unknowns, node->children[1], minimize, LoopIDToNumIterationsMap);
          // handle this node
          // create a new LLVM value (Instruction) with the operation in question
          // it will take the left and right children as operands
          // return the new LLVM value
          auto cn = insertComputationNodeAdvanced(Location, Left, Right, node->op);
          return cn;
      } else { // single child operations
          llvm::Value *Child = insertTreeEvaluationCodeAdvanced(
                  Location, CI, Unknowns, node->children[0], minimize, LoopIDToNumIterationsMap);
          return Child;
      }
    }
    return nullptr;
  }

  ExprTreeNode *locateNodeWithParticularExprTreeOp(ExprTreeNode *Node,
                                                   ExprTreeOp Op) {
    if (Node == nullptr) {
      return nullptr;
    }
    // Paimon alone knows why this terminal check is needed
    if (isTerminal(Node)) {
      if (Node->op == Op) {
        return Node;
      }
      return nullptr;
    }
    if (Node->op == Op) {
      return Node;
    }
    ExprTreeNode *Left =
        locateNodeWithParticularExprTreeOp(Node->children[0], Op);
    ExprTreeNode *Right =
        locateNodeWithParticularExprTreeOp(Node->children[1], Op);
    if (Left != nullptr) {
      return Left;
    }
    if (Right != nullptr) {
      return Right;
    }
    return nullptr;
  }

  void *collectNodesWithParticularExprTreeOp(ExprTreeNode *Node,
                                                   ExprTreeOp Op, std::vector<ExprTreeNode*> &Collection) {
    if (Node == nullptr) {
      return nullptr;
    }
    if (Node->op == Op) {
        Collection.push_back(Node);
    }
        collectNodesWithParticularExprTreeOp(Node->children[0], Op, Collection);
        collectNodesWithParticularExprTreeOp(Node->children[1], Op, Collection);
    return nullptr;
  }

  llvm::Value *insertCodeToComputePartDiff_bidx(Instruction* Location, CallBase *CI, Value *Allocation,
          ExprTreeNode *Node) {
      std::map<ExprTreeNode *, Value *> Unknowns;
      identifyUnknownsFromExpressionTree(Location, CI, Unknowns, Node);
      errs() << "unknows at partdiff \n";
      for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
              UnknownIter++) {
          errs() << (*UnknownIter).first->original_str << " ";
          (*UnknownIter).second->dump();
      }
      // locate the node in the expression tree which corresponds to the BIDX
      /* ExprTreeNode *BIDXNode = locateNodeWithParticularExprTreeOp(Node, ETO_BIDX); */
      std::vector<ExprTreeNode*> Collection;
      collectNodesWithParticularExprTreeOp(Node, ETO_BIDX, Collection);
      ExprTreeNode *BIDXNode = nullptr;
      if(!Collection.empty()){
          BIDXNode = Collection[0];
      }
          auto SumAccumulator = insertConstantNode(Location, (unsigned)0);
          SumAccumulator = insertCodeToCastInt32ToInt64(Location, SumAccumulator);
      // iteratete over all BID.x
      for(auto BIDXNodeI = Collection.begin(); BIDXNodeI != Collection.end(); BIDXNodeI++) {
          BIDXNode = *BIDXNodeI;
          if (BIDXNode == nullptr) {
              errs() << "\nNO BIDX node \n";
              return insertConstantNode(Location, (unsigned long long)(0));
          } else {
              errs() << "\nBIDX node \n";
          }
          // traverse up the tree to find multipliers
          std::vector<ExprTreeNode *> Multipliers =
              findMultipliersByTraversingUpExprTree(Node, BIDXNode);
          errs() << Multipliers.size() << "\n";
          errs() << "multipliers => ";
          std::vector<Value *> MultiplierInCode;
          for (auto Multiplier = Multipliers.begin(); Multiplier != Multipliers.end();
                  Multiplier++) {
              errs() << (*Multiplier)->original_str << ".";
              llvm::Value *Result =
                  insertTreeEvaluationCode(Location, CI, Unknowns, *Multiplier);
              if ((*Multiplier)->parent->op == ETO_SHL) {
                  Result = insertCodeToShift1By(Location, Result);
              }
              MultiplierInCode.push_back(Result);
          }
          std::vector<ExprTreeNode *> Divisions =
              findDivisorsByTraversingUpExprTree(Node, BIDXNode);
          errs() << Divisions.size() << "\n";
          errs() << "division => ";
          std::vector<Value *> DivisionInCode;
          for (auto Divisor = Divisions.begin(); Divisor != Divisions.end();
                  Divisor++) {
              errs() << (*Divisor)->original_str << ".";
              llvm::Value *Result =
                  insertTreeEvaluationCode(Location, CI, Unknowns, *Divisor);
              DivisionInCode.push_back(Result);
          }
          errs() << "\n";
          auto Accumulator = insertConstantNode(Location, (unsigned)1);
          for (auto Multiplier = MultiplierInCode.begin();
               Multiplier != MultiplierInCode.end(); Multiplier++) {
            (*Multiplier)->dump();
            Accumulator =
                insertComputationNode(Location, Accumulator, *Multiplier, ETO_MUL);
          }
          for (auto Divisor = DivisionInCode.begin();
               Divisor != DivisionInCode.end(); Divisor++) {
            (*Divisor)->dump();
            Accumulator =
                insertComputationNode(Location, Accumulator, *Divisor, ETO_DIV);
          }
          Accumulator = insertCodeToCastInt32ToInt64(Location, Accumulator);
          SumAccumulator = insertComputationNode(Location, SumAccumulator, Accumulator, ETO_ADD);
      }
      /* errs() << "\n"; */
      /* auto Accumulator = insertConstantNode(Location, (unsigned)1); */
      /* for (auto Multiplier = MultiplierInCode.begin(); */
      /*      Multiplier != MultiplierInCode.end(); Multiplier++) { */
      /*   (*Multiplier)->dump(); */
      /*   Accumulator = */
      /*       insertComputationNode(Location, Accumulator, *Multiplier, ETO_MUL); */
      /* } */
      /* Accumulator = insertCodeToCastInt32ToInt64(Location, Accumulator); */
      /* // insertCodeToPrintGenericInt64(CI, Accumulator); */
      /* insertCodeToAddPDBIDX(Location, Allocation, Accumulator); */
      /* // llvm::Value *Result = insertTreeEvaluationCode(CI, Unknowns, Node); */
      /* // return Result; */
      return SumAccumulator;
      /* return insertConstantNode(Location, (unsigned long long)(0)); */
  }

  llvm::Value *insertCodeToComputePartDiff_bidy(Instruction* Location, CallBase *CI, Value *Allocation,
                                                ExprTreeNode *Node) {
    std::map<ExprTreeNode *, Value *> Unknowns;
    identifyUnknownsFromExpressionTree(Location, CI, Unknowns, Node);
    errs() << "unknows at partdiff \n";
    for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
         UnknownIter++) {
      errs() << (*UnknownIter).first->original_str << " ";
      (*UnknownIter).second->dump();
    }
    // locate the node in the expression tree which corresponds to the BIDY
    ExprTreeNode *BIDYNode = locateNodeWithParticularExprTreeOp(Node, ETO_BIDY);
    if (BIDYNode == nullptr) {
      errs() << "NO BIDY node \n";
      return insertConstantNode(Location, (unsigned long long)(0));
    } else {
      errs() << "BIDY node \n";
    }
    // traverse up the tree to find multipliers
    std::vector<ExprTreeNode *> Multipliers =
        findMultipliersByTraversingUpExprTree(Node, BIDYNode);
    std::vector<Value *> MultiplierInCode;
    errs() << "multipliers => ";
    for (auto Multiplier = Multipliers.begin(); Multiplier != Multipliers.end();
         Multiplier++) {
      errs() << (*Multiplier)->original_str << ".";
      llvm::Value *Result =
          insertTreeEvaluationCode(Location, CI, Unknowns, *Multiplier);
      if ((*Multiplier)->parent->op == ETO_SHL) {
        Result = insertCodeToShift1By(Location, Result);
      }
      MultiplierInCode.push_back(Result);
    }
    errs() << "\n";
    auto Accumulator = insertConstantNode(Location, (unsigned)1);
    for (auto Multiplier = MultiplierInCode.begin();
         Multiplier != MultiplierInCode.end(); Multiplier++) {
      (*Multiplier)->dump();
      Accumulator =
          insertComputationNode(Location, Accumulator, *Multiplier, ETO_MUL);
    }
    Accumulator = insertCodeToCastInt32ToInt64(Location, Accumulator);
    // insertCodeToPrintGenericInt64(CI, Accumulator);
    insertCodeToAddPDBIDY(Location, Allocation, Accumulator);
    // llvm::Value *Result = insertTreeEvaluationCode(CI, Unknowns, Node);
    // return Result;
    return Accumulator;
  }

  llvm::Value *insertCodeToComputePartDiff_phi(Instruction* Location, CallBase *CI, Value *Allocation,
                                               ExprTreeNode *Node) {
    std::map<ExprTreeNode *, Value *> Unknowns;
    identifyUnknownsFromExpressionTree(Location, CI, Unknowns, Node);
    errs() << "unknows at partdiff phi \n";
    for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
         UnknownIter++) {
      errs() << (*UnknownIter).first->original_str << " ";
      (*UnknownIter).second->dump();
    }
    // locate the node in the expression tree which corresponds to the PHI
    ExprTreeNode *Phi = locateNodeWithParticularExprTreeOp(Node, ETO_PHI);
    ExprTreeNode *PhiTerm =
        locateNodeWithParticularExprTreeOp(Node, ETO_PHI_TERM);
    if (Phi == nullptr || PhiTerm == nullptr) {
      errs() << "NO PHI node \n";
      return insertConstantNode(Location, (unsigned long long)(0));
    }
    errs() << "PHI node \n";
    std::vector<ExprTreeNode *> Multipliers =
        findMultipliersByTraversingUpExprTree(Node, Phi);
    std::vector<Value *> MultiplierInCode;
    errs() << "multipliers => ";
    for (auto Multiplier = Multipliers.begin(); Multiplier != Multipliers.end();
         Multiplier++) {
      errs() << (*Multiplier)->original_str << ".";
      llvm::Value *Result =
          insertTreeEvaluationCode(Location, CI, Unknowns, *Multiplier);
      if ((*Multiplier)->parent->op == ETO_SHL) {
        Result = insertCodeToShift1By(Location, Result);
      }
      MultiplierInCode.push_back(Result);
    }
    errs() << "\n";
    auto Accumulator = insertConstantNode(Location, (unsigned)1);
    errs() << "MultiplierInCode" << MultiplierInCode.size() << "\n";  
    for (auto Multiplier = MultiplierInCode.begin();
         Multiplier != MultiplierInCode.end(); Multiplier++) {
      (*Multiplier)->dump();
      Accumulator =
          insertComputationNode(Location, Accumulator, *Multiplier, ETO_MUL);
    }
    // TODO: consider loop stride by moving from phi_term to phi
    ExprTreeNode *Current = PhiTerm;
    ExprTreeNode *Parent;
    std::vector<ExprTreeNode *> Adders;
    while (Current != Phi) {
      Parent = Current->parent;
      if (Parent == Phi) {
        break;
      }
      assert(Parent->op == ETO_ADD);
      if (Parent->children[0] == Current) {
        Adders.push_back(Parent->children[1]);
      }
      if (Parent->children[1] == Current) {
        Adders.push_back(Parent->children[0]);
      }
      Current = Parent;
    }
    auto PhiAdd = insertConstantNode(Location, unsigned(0));
    for (auto Adder = Adders.begin(); Adder != Adders.end(); Adder++) {
      errs() << "adder => " << (*Adder)->original_str << "\n";
      llvm::Value *Result = insertTreeEvaluationCode(Location, CI, Unknowns, *Adder);
      PhiAdd = insertComputationNode(Location, PhiAdd, Result, ETO_ADD);
    }
    errs() << "phiadd\n";
    PhiAdd->dump();
    Accumulator = insertCodeToCastInt32ToInt64(Location, Accumulator);
    PhiAdd = insertCodeToCastInt32ToInt64(Location, PhiAdd);
    Accumulator = insertComputationNode(Location, Accumulator, PhiAdd, ETO_MUL);
    Accumulator->dump();
    // insertCodeToPrintGenericInt64(CI, Accumulator);
    insertCodeToAddPDPhi(Location, Allocation, Accumulator);
    return Accumulator;
  }

  Value *insertCodeToGetAccessCount(CallBase *CI, Value *P) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee ACForAllocation = F->getParent()->getOrInsertFunction(
        "accessCountForAllocation", Type::getInt64Ty(Ctx),
        Type::getInt64Ty(Ctx));
    llvm::CallInst *AC = Builder.CreateCall(ACForAllocation, Args);
    // insertCodeToPrintGenericInt64(CI, AC);
    return AC;
  }

  Value *insertCodeToGetMemorySize(CallBase *CI, Value *P) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee GetAllocationSize =
        F->getParent()->getOrInsertFunction(
            "getAllocationSize", Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx));
    llvm::CallInst *Size = Builder.CreateCall(GetAllocationSize, Args);
    // insertCodeToPrintGenericInt64(CI, Size);
    return Size;
  }

  Value *insertCodeToGetAccessDensity(CallBase *CI, Value *P) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(CI);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee GetAccessDensity = F->getParent()->getOrInsertFunction(
        "getAccessDensity", Type::getFloatTy(Ctx), Type::getInt64Ty(Ctx));
    llvm::CallInst *AccessDensity = Builder.CreateCall(GetAccessDensity, Args);
    insertCodeToPrintGenericFloat32(CI, AccessDensity);
    return AccessDensity;
  }

  // Add code which will shift 1 by a 32 bit integer
  // of course, this is a trick to handle shl as multiplication
  Value *insertCodeToShift1By(Instruction *Location, Value *V) {
    assert(V->getType()->isIntegerTy(32));
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    return Builder.CreateShl(Builder.getInt32(1), V);
  }

  // Add code which will cast a 32 bit integer to 64 bit integer
  Value *insertCodeToCastInt32ToInt64(Instruction *Location, Value *V) {
    assert(V->getType()->isIntegerTy(32));
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    return Builder.CreateIntCast(V, Type::getInt64Ty(Ctx), true);
  }

  // i64 * i64 => i64
  Value* insertCodeToMultiplyInt64(Instruction* Location, Value* V1, Value* V2) {
      Value* R1 = V1;
      Value* R2 = V2;
      if(V1->getType()->isIntegerTy(32)) {
          R1 = insertCodeToCastInt32ToInt64(Location, V1);
      }
      if(V2->getType()->isIntegerTy(32)) {
          R2 = insertCodeToCastInt32ToInt64(Location, V2);
      }
      Function *F = Location->getParent()->getParent();
      LLVMContext &Ctx = F->getContext();
      IRBuilder<> Builder(Location);
      Builder.SetInsertPoint(Location);
      return Builder.CreateMul(R1, R2);
  }

  bool containsGivenArgOp(ExprTreeNode *Root, unsigned argnum) {
    if (Root == nullptr) {
      return false;
    }
    if (Root->op == ETO_ARG) {
      if (Root->arg == argnum) {
        return true;
      }
    }
    return containsGivenArgOp(Root->children[0], argnum) ||
           containsGivenArgOp(Root->children[1], argnum);
  }

  // get the enclosing loop for a given kernel invocation
  // does every loop have a preheader?
  // if a loop doesn't have a preheader, then use predecessor of header
  Instruction *getEnclosingLoopPreheaderFirst(Instruction *CI) {
    // Location->dump();
    llvm::Instruction *Loop = KernelInvocationToEnclosingLoopPredMap[CI];
    // Loop->dump();
    return Loop;
  }

  // get the enclosing loop for a given kernel invocation
  // does every loop have a preheader?
  // if a loop doesn't have a preheader, then use predecessor of header
  Instruction *getEnclosingLoopInductionVariable(Instruction *CI) {
    // Location->dump();
    llvm::Instruction *Loop = KernelInvocationToEnclosingLIVMap[CI];
    // Loop->dump();
    return Loop;
  }

  bool gridSizeIsIterationIndependent(CallBase *CI) {
    auto *KernelPointer = CI->getArgOperand(0);
    auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
    auto KernelName = KernelFunction->getName();
    std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
    auto LIV = KernelInvocationToEnclosingLIVMap[CI];
    if (LIV == nullptr) {
      errs() << "PANIC: no enclosing loop found for kernel invocation\n";
      return false;
    }
    unsigned LoopArg = KernelInvocationToLIVToArgNumMap[CI][LIV];
    auto *GridDimXValue =
        KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
    Value *gridDimXValue = nullptr;
    if (GridDimXValue == nullptr) {
      llvm::Value *gdimxy_value = KernelInvocationToGridDimXYValueMap[CI];
      if (gdimxy_value) {
        assert(gdimxy_value->getType()->isIntegerTy(64));
        errs() << "gdimxy value found\n";
        Function *F = CI->getParent()->getParent();
        LLVMContext &Ctx = F->getContext();
        IRBuilder<> Builder(CI);
        auto *GridDimXValue_64_a = Builder.CreateShl(gdimxy_value, 32);
        auto *GridDimXValue_64 = Builder.CreateLShr(GridDimXValue_64_a, 32);
        auto *GridDimXValue_32 = Builder.CreateIntCast(
            GridDimXValue_64, Type::getInt32Ty(Ctx), false);
        gridDimXValue = GridDimXValue_32;
      }
    } else {
      gridDimXValue = GridDimXValue;
    }
    assert(gridDimXValue != nullptr);
    // traverse gridDimXValue to see if it contains LIV
    if (isDependentOn(gridDimXValue, LIV)) {
      errs() << "gridDimXValue is dependent on LIV\n";
      return false;
    }
    return true;
  }

  // the following function checks if an LLVM value is dependent on another LLVM
  bool isDependentOn(llvm::Value *V, llvm::Value *W) {
    if (V == W) {
      return true;
    }
    auto *I = dyn_cast<Instruction>(V);
    if (I) {
      for (auto &U : I->operands()) {
        if (isDependentOn(U, W)) {
          return true;
        }
      }
    }
    return false;
  }

  void identifyIterationDependentAccesses(
      Instruction *Location, CallBase *CI,
      std::map<unsigned, Value *> LoopIDToNumIterationsMap) {
    errs() << "identify iteration dependent accesses\n";
    auto *KernelPointer = CI->getArgOperand(0);
    auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
    auto KernelName = KernelFunction->getName();
    std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
    std::map<unsigned, ExprTreeNode *> AccessIDToExprMap =
        KernelNameToAccessIDToExpressionTreeMap[OriginalKernelName];
    // get the host side loop induction variable
    auto LIV = KernelInvocationToEnclosingLIVMap[CI];
    if (LIV == nullptr) {
      errs() << "PANIC: no enclosing loop found for kernel invocation\n";
      return;
    }
    // looparg is the host side LIV that is used in the kernel for some
    // computation
    unsigned LoopArg = KernelInvocationToLIVToArgNumMap[CI][LIV];
    errs() << "loop arg is " << LoopArg << "\n";
    for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end();
         AID++) {
      auto Expr = (*AID).second;
      if (containsGivenArgOp(Expr, LoopArg)) {
        errs() << "access id " << (*AID).first << " is dependent on loop arg "
               << LoopArg << "\n";
        // need max grid dim, block dim, and loop bound
        std::map<unsigned, unsigned> AccessIDToLoopIDMap =
            KernelNameToAccessIDToEnclosingLoopMap[OriginalKernelName];
        auto LoopID = AccessIDToLoopIDMap[AID->first];
        llvm::Value *LoopIters = LoopIDToNumIterationsMap[LoopID];
        auto wss = estimateWorkingSetSizeIteration(Location, CI, (*AID).second,
                                                   LoopArg, LoopIters);
        /* insertCodeToPrintGenericInt32(Location, wss); */
        insertCodeToAddWSS_iterdep(Location, AID->first, wss);
        // insertCodeToAddAccessCountPerAccess(Location, AID->first, ac);
      }
    }
    insertCodeToProcessWSS_iterdep(Location);
  }

  void identifyMinForUnknows(Instruction *Location, CallBase *CI,
                             std::map<ExprTreeNode *, Value *> &Unknowns,
                             ExprTreeNode *node) {
    if (node == nullptr)
      return;
    if (isTerminal(node)) {
      if (node->op == ETO_TIDX) {
        auto unknown = insertConstantNode(Location, unsigned(0));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_TIDY) {
        auto unknown = insertConstantNode(Location, unsigned(0));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDX) {
        auto unknown = insertConstantNode(Location, unsigned(0));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDY) {
        auto unknown = insertConstantNode(Location, unsigned(0));
        Unknowns[node] = (unknown);
        return;
      }
      // Assertion will only work if all cases are handled.
      // assert(false);
    } else {
      if (node->op == ETO_PHI) {
        errs() << "PHI TERM\n";
        return;
      }
      /* errs() << "unknown check? not terminal\n" << "\n"; */
      identifyMinForUnknows(Location, CI, Unknowns, node->children[0]);
      identifyMinForUnknows(Location, CI, Unknowns, node->children[1]);
    }

    return;
  }

  // This function considers BIDs to be constant.
  // TODO: change the function to take list of arguments to be treated constant.
  void identifyMinForUnknowsAdvanced(Instruction *Location, CallBase *CI,
                             std::map<ExprTreeNodeAdvanced *, Value *> &Unknowns,
                             ExprTreeNodeAdvanced *node) {
    if (node == nullptr)
      return;
    errs() << "id min for " << node->original_str << "\n";
    if (isTerminal(node)) {
      if (node->op == ETO_TIDX) {
        auto unknown = insertConstantNode(Location, unsigned(0));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_TIDY) {
        auto unknown = insertConstantNode(Location, unsigned(0));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDX) {
        auto unknown = insertConstantNode(Location, (4));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDY) {
        auto unknown = insertConstantNode(Location, (4));
        Unknowns[node] = (unknown);
        return;
      }
      // Assertion will only work if all cases are handled.
      // assert(false);
    } else {
      if (node->op == ETO_PHI) {
        errs() << "PHI TERM\n";
            for(auto child = node->children.begin(); child != node->children.end(); child++) {
                identifyMinForUnknowsAdvanced(Location, CI, Unknowns, *child);
            }
        return;
      }
      /* errs() << "unknown check? not terminal\n" << "\n"; */
      for(auto child = node->children.begin(); child != node->children.end(); child++) {
          identifyMinForUnknowsAdvanced(Location, CI, Unknowns, *child);
      }
    }

    return;
  }

  void identifyMaxForUnknows(Instruction *Location, CallBase *CI,
                             std::map<ExprTreeNode *, Value *> &Unknowns,
                             ExprTreeNode *node) {
    if (node == nullptr)
      return;
    if (isTerminal(node)) {
      if (node->op == ETO_TIDX) {
        unsigned bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
        /* unknown->dump(); */
        auto unknown = insertConstantNode(Location, bdimx - 1);
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_TIDY) {
        unsigned bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
        /* unknown->dump(); */
        auto unknown = insertConstantNode(Location, bdimy - 1);
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDX) {
        auto gdimx_value =
            KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
        if (gdimx_value) {
          errs() << "gdimx value found\n";
          gdimx_value->dump();
          Unknowns[node] = gdimx_value - 1;
          return;
        }
        llvm::Value *gdimxy_value = KernelInvocationToGridDimXYValueMap[CI];
        if (gdimxy_value) {
          assert(gdimxy_value->getType()->isIntegerTy(64));
          errs() << "gdimxy value found\n";
          Function *F = CI->getParent()->getParent();
          LLVMContext &Ctx = F->getContext();
          IRBuilder<> Builder(Location);
          auto *GridDimXValue_64_a = Builder.CreateShl(gdimxy_value, 32);
          auto *GridDimXValue_64 = Builder.CreateLShr(GridDimXValue_64_a, 32);
          auto *GridDimXValue_32 = Builder.CreateIntCast(
              GridDimXValue_64, Type::getInt32Ty(Ctx), false);
          auto one = insertConstantNode(Location, (unsigned)1);
          auto unk =
              insertComputationNode(Location, GridDimXValue_32, one, ETO_SUB);
          Unknowns[node] = unk;
          return;
        }
        unsigned gdimx = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX];
        errs() << "Gridm is " << gdimx << "\n";
        /* unknown->dump(); */
        auto unknown = insertConstantNode(Location, gdimx - 1);
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDY) {
        auto gdimy_value =
            KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY];
        if (gdimy_value) {
          errs() << "gdimy value found\n";
          gdimy_value->dump();
          Unknowns[node] = gdimy_value - 1;
          return;
        }
        llvm::Value *gdimxy_value = KernelInvocationToGridDimXYValueMap[CI];
        if (gdimxy_value) {
          assert(gdimxy_value->getType()->isIntegerTy(64));
          errs() << "gdimxy value found\n";
          Function *F = CI->getParent()->getParent();
          LLVMContext &Ctx = F->getContext();
          IRBuilder<> Builder(Location);
          auto *GridDimYValue_64 = Builder.CreateLShr(gdimxy_value, 32);
          auto *GridDimYValue_32 = Builder.CreateIntCast(
              GridDimYValue_64, Type::getInt32Ty(Ctx), false);
          auto one = insertConstantNode(Location, (unsigned)1);
          auto unk =
              insertComputationNode(Location, GridDimYValue_32, one, ETO_SUB);
          Unknowns[node] = unk;
          return;
        }
        unsigned gdimy = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY];
        errs() << "Gridm is " << gdimy << "\n";
        /* unknown->dump(); */
        auto unknown = insertConstantNode(Location, gdimy - 1);
        Unknowns[node] = (unknown);
        return;
      }
      // Assertion will only work if all cases are handled.
      // assert(false);
    } else {
      if (node->op == ETO_PHI) {
        errs() << "PHI TERM haha not handled\n";
        return;
      }
      /* errs() << "unknown check? not terminal\n" << "\n"; */
      identifyMaxForUnknows(Location, CI, Unknowns, node->children[0]);
      identifyMaxForUnknows(Location, CI, Unknowns, node->children[1]);
    }

    return;
  }

  // This function considers BIDs to be constant.
  // TODO: change the function to take list of arguments to be treated constant.
  void identifyMaxForUnknowsAdvanced(Instruction *Location, CallBase *CI,
                             std::map<ExprTreeNodeAdvanced *, Value *> &Unknowns,
                             ExprTreeNodeAdvanced *node) {
    if (node == nullptr)
      return;
    errs() << "id max for " << node->original_str << "\n";
    if (isTerminal(node)) {
      if (node->op == ETO_TIDX) {
        unsigned bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
        /* unknown->dump(); */
        auto unknown = insertConstantNode(Location, bdimx - 1);
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_TIDY) {
        unsigned bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
        /* unknown->dump(); */
        auto unknown = insertConstantNode(Location, bdimy - 1);
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDX) {
        auto unknown = insertConstantNode(Location, unsigned(4));
        Unknowns[node] = (unknown);
        return;
      }
      if (node->op == ETO_BIDY) {
        auto unknown = insertConstantNode(Location, unsigned(4));
        Unknowns[node] = (unknown);
        return;
      }
      // Assertion will only work if all cases are handled.
      // assert(false);
    } else {
        if (node->op == ETO_PHI) {
            errs() << "PHI TERM haha not handled\n";
            for(auto child = node->children.begin(); child != node->children.end(); child++) {
                identifyMaxForUnknowsAdvanced(Location, CI, Unknowns, *child);
            }
            return;
        }
      /* errs() << "unknown check? not terminal\n" << "\n"; */
      for(auto child = node->children.begin(); child != node->children.end(); child++) {
          identifyMaxForUnknowsAdvanced(Location, CI, Unknowns, *child);
      }
    }

    return;
  }

  Value *
  insertCodeToEstimateMaxValue(Instruction *Location, CallBase *CI,
                               ExprTreeNode *Node,
                               std::map<ExprTreeNode *, Value *> Unknowns,
                               unsigned LoopArg, Value *LoopIters) {
    // TODO: add bidx, bidy, tidx, tidy etc to the unknowns
    // for example, bidx = gridDimX - 1 // since max
    // also tidx = blockDimX - 1 // since max
      errs() << "insert code to estimate max value\n";
    identifyMaxForUnknows(Location, CI, Unknowns, Node);
    return insertTreeEvaluationCode(Location, CI, Unknowns, Node, LoopIters);
    // return nullptr;
  }

  // This function considers BIDs to be constant.
  // TODO: change the function to take list of arguments to be treated constant.
  Value*
  insertCodeToEstimateMaxValueAdvanced(Instruction *Location, CallBase *CI,
                               ExprTreeNodeAdvanced *Node,
                               std::map<ExprTreeNodeAdvanced *, Value *> Unknowns,
                               unsigned LoopArg, Value *LoopIters,
                               std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
    // TODO: add bidx, bidy, tidx, tidy etc to the unknowns
    // for example, bidx = gridDimX - 1 // since max
    // also tidx = blockDimX - 1 // since max
      errs() << "insert code to estimate max value\n";
    identifyMaxForUnknowsAdvanced(Location, CI, Unknowns, Node);
    return insertTreeEvaluationCodeAdvanced(Location, CI, Unknowns, Node, false, LoopIDToNumIterationsMap);
    // return nullptr;
  }

  Value *insertCodeToEstimateMinValue(
      Instruction *Location, CallBase *CI, ExprTreeNode *Node,
      std::map<ExprTreeNode *, Value *> Unknowns, unsigned LoopArg) {
      errs() << "insert code to estimate min value\n";
    identifyMinForUnknows(Location, CI, Unknowns, Node);
    auto zero = insertConstantNode(Location, (unsigned)0);
    return insertTreeEvaluationCode(Location, CI, Unknowns, Node, zero);
  }

  Value *insertCodeToEstimateMinValueAdvanced(
      Instruction *Location, CallBase *CI, ExprTreeNodeAdvanced *Node,
      std::map<ExprTreeNodeAdvanced *, Value *> Unknowns, unsigned LoopArg, std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
      errs() << "insert code to estimate min value\n";
    identifyMinForUnknowsAdvanced(Location, CI, Unknowns, Node);
    auto zero = insertConstantNode(Location, (unsigned)0);
    return insertTreeEvaluationCodeAdvanced(Location, CI, Unknowns, Node, true, LoopIDToNumIterationsMap);  // true for minimize. change to enum
  }

  Value *estimateWorkingSetSizeIteration(Instruction *Location, CallBase *CI,
                                         ExprTreeNode *Node, unsigned LoopArg,
                                         Value *LoopIters) {
    // get the max value the expression tree can take in an iteration
    // get the min value the expression tree can take in an iteration
    // get the difference
    std::map<ExprTreeNode *, Value *> Unknowns;
    identifyUnknownsFromExpressionTree(Location, CI, Unknowns, Node);
    errs() << "\nID unknowns\n";
    for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
         UnknownIter++) {
      (*UnknownIter).second->dump();
    }
    Value *maxValue = insertCodeToEstimateMaxValue(Location, CI, Node, Unknowns,
                                                   LoopArg, LoopIters);
    // insertCodeToPrintGenericInt32(Location, maxValue);
    Value *minValue =
        insertCodeToEstimateMinValue(Location, CI, Node, Unknowns, LoopArg);
    // insertCodeToPrintGenericInt32(Location, minValue);
    // assert(maxValue && maxValue->getType()->isIntegerTy(64));
    // assert(minValue && minValue->getType()->isIntegerTy(64));
    llvm::Value *wss =
        insertComputationNode(Location, maxValue, minValue, ETO_SUB);
    // insertCodeToPrintGenericInt32(Location, wss);
    return wss;
    // return nullptr;
  }

  bool isTrivialExpression(Instruction* Location, CallBase* CI, ExprTreeNodeAdvanced* Node) {
      // traverse the expression tree, and check for GEPs
      // If there is no gep, then there are no indices to examine
      if(detectParticularNode(Node, ETO_GEP) == false) {
          return true;
      }
      return false;
  }

  Value *estimateWorkingSetSizeAdvanced(Instruction *Location, CallBase *CI,
          ExprTreeNodeAdvanced *Node, Value* LoopIters,
          Value *BDIMX, Value* BDIMY,
          std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
      // get the max value the expression tree can take in an iteration
      // get the min value the expression tree can take in an iteration
      // get the difference
      std::map<ExprTreeNodeAdvanced*, Value *> Unknowns;
      // first though, check if expression is trivial
      if(isTrivialExpression(Location, CI, Node)) {
          errs() << "Trivial expression\n";
          return insertConstantNode(Location, (unsigned long long)(1));
      }
      identifyUnknownsFromExpressionTreeAdvanced(Location, CI, Unknowns, Node);
      errs() << "\nID unknowns\n";
      for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
              UnknownIter++) {
          (*UnknownIter).second->dump();
      }
      Value *maxValue = insertCodeToEstimateMaxValueAdvanced(Location, CI, Node, Unknowns,
              0, LoopIters, LoopIDToNumIterationsMap);
      maxValue = insertCodeToCastInt32ToInt64(Location, maxValue);
      Value *minValue =
          insertCodeToEstimateMinValueAdvanced(Location, CI, Node, Unknowns, 0, LoopIDToNumIterationsMap);
      minValue = insertCodeToCastInt32ToInt64(Location, minValue);
      insertCodeToPrintGenericInt64(Location, maxValue);
      insertCodeToPrintGenericInt64(Location, minValue);
      auto deadbeef = insertConstantNode(Location, unsigned(42042));
      insertCodeToPrintGenericInt32(Location, deadbeef);
      // assert(maxValue && maxValue->getType()->isIntegerTy(64));
      // assert(minValue && minValue->getType()->isIntegerTy(64));
      llvm::Value *wss =
          insertComputationNode(Location, maxValue, minValue, ETO_SUB);
      if(wss->getType()->isIntegerTy(32)) {
          wss = insertCodeToCastInt32ToInt64(Location, wss);
      }
      // insertCodeToPrintGenericInt32(Location, wss);
      errs() << "estimating working set size\n";
      Function *F = Location->getParent()->getParent();
      LLVMContext &Ctx = F->getContext();
      IRBuilder<> Builder(Location);
      /* V->getType()->dump(); */
      /* llvm::Value *Ptr = Builder.CreatePtrToInt(Pointer, Builder.getInt64Ty()); */
      Value *Args[] = {wss, BDIMX, BDIMY};

    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee estimateWorkingSet =
        F->getParent()->getOrInsertFunction(
            "estimate_working_set2", Type::getInt64Ty(Ctx),
            Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    llvm::CallInst *wssR = Builder.CreateCall(estimateWorkingSet, Args);
    insertCodeToPrintGenericInt64(Location, wssR);
    return wssR;
    // return nullptr;
  }
  // Same name, different signature.
  // Estimates working set size of threadblock
  Value *estimateWorkingSetSize(Instruction *Location, CallBase *CI,
                                         ExprTreeNode *Node, Value* LoopIters, Value *BDIMX, Value* BDIMY) {
    // get the max value the expression tree can take in an iteration
    // get the min value the expression tree can take in an iteration
    // get the difference
    std::map<ExprTreeNode *, Value *> Unknowns;
    identifyUnknownsFromExpressionTree(Location, CI, Unknowns, Node);
    errs() << "\nID unknowns\n";
    for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
         UnknownIter++) {
      (*UnknownIter).second->dump();
    }
    Value *maxValue = insertCodeToEstimateMaxValue(Location, CI, Node, Unknowns,
                                                   0, LoopIters);
    // insertCodeToPrintGenericInt32(Location, maxValue);
    Value *minValue =
        insertCodeToEstimateMinValue(Location, CI, Node, Unknowns, 0);
    // insertCodeToPrintGenericInt32(Location, minValue);
    // assert(maxValue && maxValue->getType()->isIntegerTy(64));
    // assert(minValue && minValue->getType()->isIntegerTy(64));
    llvm::Value *wss =
        insertComputationNode(Location, maxValue, minValue, ETO_SUB);
    if(wss->getType()->isIntegerTy(32)) {
        wss = insertCodeToCastInt32ToInt64(Location, wss);
    }
    // insertCodeToPrintGenericInt32(Location, wss);
    errs() << "estimating working set size\n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    /* llvm::Value *Ptr = Builder.CreatePtrToInt(Pointer, Builder.getInt64Ty()); */
    Value *Args[] = {wss, BDIMX, BDIMY};
                    
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee estimateWorkingSet =
        F->getParent()->getOrInsertFunction(
            "estimate_working_set2", Type::getInt64Ty(Ctx),
            Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    llvm::CallInst *wssR = Builder.CreateCall(estimateWorkingSet, Args);
    insertCodeToPrintGenericInt64(Location, wssR);
    return wssR;
    // return nullptr;
  }

  // num threads is the number of threads in the kernel, i.e. the product of
  // the number of threads in each threadblock (blockdim) multiplied by the
  // number of threadblocks in the grid (griddim)
  Value *insertCodeToPrintNumThreads(Instruction* Location,
      CallBase *CI, std::map<CallBase *, Value *> &KernelInvocationToBDimXMap,
      std::map<CallBase *, Value *> &KernelInvocationToBDimYMap,
      std::map<CallBase *, Value *> &KernelInvocationToGDimXMap,
      std::map<CallBase *, Value *> &KernelInvocationToGDimYMap) {
    Function *F = CI->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    if (KernelInvocationToGridDimXYValueMap.find(CI) ==
        KernelInvocationToGridDimXYValueMap.end()) {
      errs() << "PANIC: no grid dim XY value found\n";
      return nullptr;
    }
    if (KernelInvocationToGridDimZValueMap.find(CI) ==
        KernelInvocationToGridDimZValueMap.end()) {
      errs() << "PANIC: no grid dim Z value found\n";
      return nullptr;
    }
    Value *GridDimXYValue = KernelInvocationToGridDimXYValueMap[CI];
    Value *GridDimZValue = KernelInvocationToGridDimZValueMap[CI];
    GridDimXYValue->dump();
    GridDimXYValue->getType()->dump();
    GridDimZValue->dump();
    GridDimZValue->getType()->dump();
    assert(GridDimXYValue->getType()->isIntegerTy(64));
    assert(GridDimZValue->getType()->isIntegerTy(32));
    // GridDimY = GridXY >> 32;
    // GridDimX = (GridXY << 32) >> 32;
    Builder.SetInsertPoint(Location);
    auto *GridDimYValue_64 = Builder.CreateLShr(GridDimXYValue, 32);
    auto *GridDimYValue_32 =
        Builder.CreateIntCast(GridDimYValue_64, Type::getInt32Ty(Ctx), false);
    auto *GridDimXValue_64_a = Builder.CreateShl(GridDimXYValue, 32);
    auto *GridDimXValue_64 = Builder.CreateLShr(GridDimXValue_64_a, 32);
    auto *GridDimXValue_32 =
        Builder.CreateIntCast(GridDimXValue_64, Type::getInt32Ty(Ctx), false);
    insertCodeToPrintGenericInt32(Location, GridDimXValue_32);
    insertCodeToPrintGenericInt32(Location, GridDimYValue_32);
    insertCodeToPrintGenericInt32(Location, GridDimZValue);
    KernelInvocationToGDimXMap[CI] = GridDimXValue_32;
    KernelInvocationToGDimYMap[CI] = GridDimYValue_32;
    auto *GridDimZValue_64 =
        Builder.CreateIntCast(GridDimZValue, Type::getInt64Ty(Ctx), false);
    auto *GridDimXYProductValue_64 =
        Builder.CreateMul(GridDimXValue_64, GridDimYValue_64);
    auto *GridDimXYZProductValue_64 =
        Builder.CreateMul(GridDimXYProductValue_64, GridDimZValue_64);
    // next we compute number of threads per threadblock
    // assuming numThreadsPerThreadBlock is a constant, and less than 2^32
    unsigned int NumThreadsPerThreadBlock =
        KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX] *
        KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY] *
        KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMZ];
    auto BlockDimXValue = insertConstantNode(
        Location, KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX]);
    auto BlockDimYValue = insertConstantNode(
        Location, KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY]);
    KernelInvocationToBDimXMap[CI] = BlockDimXValue;
    KernelInvocationToBDimYMap[CI] = BlockDimYValue;
    // insert code to multiply numThreadsPerThreadBlock with
    // GridDimXYProductValue_64
    auto *NumThreadsPerThreadBlockValue_64 =
        Builder.getInt64(NumThreadsPerThreadBlock);
    auto *NumThreadsValue_64 = Builder.CreateMul(
        GridDimXYZProductValue_64, NumThreadsPerThreadBlockValue_64);
    // insertCodeToPrintGenericInt64(CI, NumThreadsValue_64);
    return NumThreadsValue_64;
  }

  llvm::Instruction *insertCodeForIterationDecision(Instruction *Location, Instruction* LIV) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    ArrayRef<Value *> Args = {LIV};
    auto Fn = F->getParent()->getOrInsertFunction("penguinSuperPrefetchWrapper",
                                                  Type::getVoidTy(Ctx), 
                                                  Type::getInt32Ty(Ctx));
    auto *Result = Builder.CreateCall(Fn, Args);
    return Result;
  }

  // insert an block which will run on the first iteration only
  llvm::Instruction *insertCodeForFirstIterationExecution(Instruction *Location,
                                                          Value *LIV) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    // Builder.CreateCondBr(Builder.CreateICmpEQ(LIV, Builder.getInt32(0)),
    //  Builder.GetInsertBlock(), Builder.GetInsertBlock());
    // BasicBlock* FirstIterBB = BasicBlock::Create(Ctx, "first_iter");
    llvm::Instruction *IfThen = SplitBlockAndInsertIfThen(
        Builder.CreateICmpEQ(LIV, Builder.getInt32(0)), Location, false);
    Builder.SetInsertPoint(IfThen);
    // auto fortytwo = insertConstantNode(IfThen, unsigned(42));
    // insertCodeToPrintGenericInt32(IfThen, fortytwo);
    // we insert the code to perform global memory management decisions,
    // other runtime information gathering will happen BEFORE this call
    return insertCodeToPerformGlobalMemoryMgmt(IfThen);
    // return IfThen;
  }

  // Some kernel loops may be dependent on arguments to the kernel.
  // For such loops, we compute the number of iterations.
  // This will be important to compute the access density.
  // The output is a map, which contains loop id to the number of iterations.
  // Note that loop id is obtained from the output of the kernel analysis.
  void insertCodeToComputeKernelLoopIterationCount(
      Instruction *Location, CallBase *CI,
      std::map<unsigned, Value *> &LoopIDToNumIterationsMap,
      std::map<unsigned, bool> &LoopIDToIncompMap) {
    auto *KernelPointer = CI->getArgOperand(0);
    auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
    auto KernelName = KernelFunction->getName();
    std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
    std::map<unsigned, std::vector<std::string>> kernelLoopToBoundsMap =
        LoopIDToLoopBoundsMap[OriginalKernelName];
    for (auto LoopID = kernelLoopToBoundsMap.begin();
         LoopID != kernelLoopToBoundsMap.end(); LoopID++) {
      errs() << "Loop ID = " << LoopID->first << "\n";
      ExprTreeNode *LoopIters = partiallyEvaluatedLoopIters(
          Location, CI, OriginalKernelName, LoopID->first);
      if (LoopIters == nullptr) {
        errs() << "\nPANIC: serious problem with partially evaluated loop "
                  "iters\n";
        /* assert(false); */
        LoopIDToIncompMap[LoopID->first] = true;
      } else {
        std::map<ExprTreeNode *, Value *> Unknowns;
        identifyUnknownsFromExpressionTree(Location, CI, Unknowns, LoopIters);
        errs() << "\nID unknowns\n";
        for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
             UnknownIter++) {
          (*UnknownIter).second->dump();
        }
        auto ItersValue =
            insertLoopItersEvaluationCode(Location, CI, Unknowns, LoopIters);
        LoopIDToNumIterationsMap[LoopID->first] = ItersValue;
        errs() << "itersvalue = ";
        ItersValue->dump();
        insertCodeToPrintGenericInt32(Location, ItersValue);
        LoopIDToIncompMap[LoopID->first] = false;
      }
    }


    return;
  }

  void insertCodeToComputeConditionalExecutionProbability(
      Instruction *Location, CallBase *CI,
      std::map<unsigned, Value*> &IfIDToProbMap) {
    errs() << "bedug hello\n";
    for(auto IfID = IfIDToCondMap.begin();
        IfID != IfIDToCondMap.end(); IfID++) {
      errs() << "\nIF ID = " << IfID->first;
      ExprTreeNode* expr = createExpressionTree(IfID->second);
      std::map<ExprTreeNode *, Value *> Unknowns;
      identifyUnknownsFromExpressionTree(Location, CI, Unknowns, expr);
      errs() << "\nID unknowns\n";
      for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
          UnknownIter++) {
        (*UnknownIter).second->dump();
      }
        auto NumExecs =
            insertIfProbEvalCode(Location, CI, Unknowns, expr);
        errs() << "bedug hello 2\n";
        NumExecs->dump();
        insertCodeToPrintGenericFloat64(Location, NumExecs);
    }
  }

  // identify all the nested loops, sum up (multiply) the number of iterations.
  Value* insertCodeComputeLoopIterationCountNested(
          Instruction* Location, unsigned loopid,
          std::map<unsigned, Value*> LoopIDToNumIterationsMap) {
      errs() << "nested loop count evaluation. assuming loopid to num iters map is populated\n";
      Value* LoopIters = LoopIDToNumIterationsMap[loopid];
      if(LoopIters->getType()->isIntegerTy(32)) {
          LoopIters = insertCodeToCastInt32ToInt64(Location, LoopIters);
      }
      unsigned parentLoopId = LoopIDToParentLoopIDMap[loopid];
      while(parentLoopId != 0) {
          errs() << "lid = " << loopid << " pid = " << parentLoopId << "\n";
          Value* ParentLoopIters = LoopIDToNumIterationsMap[parentLoopId];
          LoopIters = insertCodeToMultiplyInt64(Location, LoopIters, ParentLoopIters);
          loopid = parentLoopId;
          parentLoopId = LoopIDToParentLoopIDMap[loopid];
      }
      return LoopIters;
  }

  Instruction* insertPointForFirstInvocationNonIter(Instruction* Location) {
      errs() << "insertPointForFirstInvocationNonIter\n:";
      Location->dump();
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* Value *Args[] = {}; */
        ArrayRef<Value *> PrintArgs = {};
    llvm::FunctionCallee AddAIDToInvocationID = F->getParent()->getOrInsertFunction(
        "MemoryMgmtFirstInvocationNonIter", Type::getVoidTy(Ctx));
    return Builder.CreateCall(AddAIDToInvocationID, PrintArgs);
    // return a special point
  }

  void insertCodeToRecordReuse(Instruction* Location, unsigned invid, unsigned aid, Value* ac, Value* Allocation) {
      errs() << "insert code to record reuse\n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    // Builder.CreateCall(Fn, Args);
    llvm::Value *AID = Builder.getInt32(aid);
    llvm::Value *INVID = Builder.getInt32(invid);
    auto FirstMap = AllocationToFirstMap[Allocation];
    if(FirstMap) {
        Allocation = FirstMap;
    }
    llvm::Value *Ptr = Builder.CreatePtrToInt(Allocation, Builder.getInt64Ty());
    /* Ptr->dump(); */

    // check if Allocation dominates Ptr
    // get the first instance of allocation
    Value *Args[] = {AID, INVID};
    llvm::FunctionCallee AddAIDToInvocationID = F->getParent()->getOrInsertFunction(
        "add_aid_invocation_map_reuse", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt32Ty(Ctx));
    Builder.CreateCall(AddAIDToInvocationID, Args);
    Value *Args2[] = {AID, Ptr};
    llvm::FunctionCallee AddAIDToAllocation = F->getParent()->getOrInsertFunction(
        "add_aid_allocation_map_reuse", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddAIDToAllocation, Args2);
    Value *Args3[] = {AID, ac};
    llvm::FunctionCallee AddAIDToAC = F->getParent()->getOrInsertFunction(
        "add_aid_ac_map_reuse", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx),
        Type::getInt64Ty(Ctx));
    Builder.CreateCall(AddAIDToAC, Args3);
  }
  // find the loop bounds for the loop with the given loop id

  void insertCodeToComputeAccessDensity(Instruction* Location,
      CallBase *CI, Value *NumThreadsInGrid,
      std::map<unsigned, Value *> LoopIDToNumIterationsMap,
      std::map<unsigned, bool> LoopIDToIncompMap,
      std::map<CallBase *, Value *> &KernelInvocationToBDimXMap,
      std::map<CallBase *, Value *> &KernelInvocationToBDimYMap,
      std::map<CallBase *, Value *> &KernelInvocationToGDimXMap,
      std::map<CallBase *, Value *> &KernelInvocationToGDimYMap) {
    errs() << "called function dynamic AD computation\n";
    // to compute access density, we need the following in real time.
    // 1. number of threads
    // 2. loop count inside the kernel
    // 3. size of data structures being accessed
    Function *F = CI->getParent()->getParent();
    CI->dump();
    auto *KernelPointer = CI->getArgOperand(0);
    auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
    auto KernelName = KernelFunction->getName();
    std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
    std::map<unsigned, unsigned> AccessIDToLoopIDMap =
        KernelNameToAccessIDToEnclosingLoopMap[OriginalKernelName];
    std::map<unsigned, unsigned> AccessIDToAllocArgMap =
        KernelNameToAccessIDToAllocationArgMap[OriginalKernelName];
    std::map<unsigned, ExprTreeNode *> AccessIDToExprMap =
        KernelNameToAccessIDToExpressionTreeMap[OriginalKernelName];
    std::map<unsigned, ExprTreeNodeAdvanced *> AccessIDToAdvancedExprMap=
        KernelNameToAccessIDToAdvancedExpressionTreeMap[OriginalKernelName];
    std::set<Value *> MallocPointerKernArgs;
    for (auto AID = AccessIDToLoopIDMap.begin();
         AID != AccessIDToLoopIDMap.end(); AID++) {
        // TODO :: add check if AID is involved with kernel invocation
      // LoopIDs start from 1, by Penguin convention
      ExprTreeNode *Expr = AccessIDToExprMap[AID->first];
      ExprTreeNodeAdvanced *AdvExpr = AccessIDToAdvancedExprMap[AID->first];
      unsigned AllocArg = AccessIDToAllocArgMap[AID->first];
      errs() << "allocation arg = " << AllocArg << "\n";
      auto Allocation =
          KernelInvocationToArgNumberToAllocationMap[CI][AllocArg];
      Allocation->dump();
      MallocPointerKernArgs.insert(Allocation);
      if(isPointerChase(Expr)) {
          // set the allocation as pointer chase
          insertCodeToSetPChase(Location, AID->first, Allocation, true);
          continue; // continue with other accesses (AID for loop).
      }
      llvm::Value *ExecutionCount;
      llvm::Value *LoopIters;
      if (AID->second == 0) {
        errs() << "\nAID = " << AID->first << " is not in a loop\n";
        LoopIters = insertConstantNode(Location, unsigned(1));
        ExecutionCount = NumThreadsInGrid;
        // insertCodeToPrintGenericInt64(Location, ExecutionCount);
      } else {
        errs() << "\nAID = " << AID->first;
        errs() << "\nLoop ID = " << AID->second;
        /* LoopIters = LoopIDToNumIterationsMap[AID->second]; // returns 32 bit */
        /* llvm::Value *LoopIters_64 = insertCodeToCastInt32ToInt64(Location, LoopIters); */
        // loop iters for nested loops
        // If loop bounds are hard to compute (i.e., unbounded), then cannot compute access density.
        if(LoopIDToIncompMap[AID->second] == true) {
            errs() << "loop is incomputable\n";
          insertCodeToSetIncomp(Location, AID->first, true);
          continue; // continue with other accesses (AID for loop).
        }
       LoopIters= insertCodeComputeLoopIterationCountNested(Location, AID->second, LoopIDToNumIterationsMap); // returns 64 bit
        auto *LoopItersTimesNumThreads =
            insertComputationNode(Location, LoopIters, NumThreadsInGrid, ETO_MUL);
        ExecutionCount = LoopItersTimesNumThreads;
        // insertCodeToPrintGenericInt64(Location, ExecutionCount);
      }
      // get the pointer to the data structure being accessed
      insertCodeToAddAccessCount(Location, AID->first, Allocation, ExecutionCount);
      // Next, we compute partial differences
      llvm::Value *PartDiff_bidx;
      PartDiff_bidx = insertCodeToComputePartDiff_bidx(Location, CI, Allocation, Expr);
      llvm::Value *PartDiff_bidy;
      PartDiff_bidy = insertCodeToComputePartDiff_bidy(Location, CI, Allocation, Expr);
      llvm::Value *PartDiff_phi;
      PartDiff_phi = insertCodeToComputePartDiff_phi(Location, CI, Allocation, Expr);
      /* auto wss = estimateWorkingSetSize( */
          /* Location, Allocation, PartDiff_bidx, PartDiff_bidy, PartDiff_phi, LoopIters, */
          /* KernelInvocationToBDimXMap[CI], KernelInvocationToBDimYMap[CI], */
          /* KernelInvocationToGDimXMap[CI], KernelInvocationToGDimYMap[CI]); */
      /* auto wss = estimateWorkingSetSize(Location, CI, Expr, LoopIters, KernelInvocationToBDimXMap[CI], KernelInvocationToBDimYMap[CI]); // the more refined wss estimation */
      // the more refined wss estimation
      if(isIndirectAccess(AdvExpr)) {
          // TODO, set as wss estimation not possible
          continue; // continue with other accesses (AID for loop).
          /* return; */
      }
      auto wss_advanced = estimateWorkingSetSizeAdvanced(Location, CI, AdvExpr, LoopIters, KernelInvocationToBDimXMap[CI], KernelInvocationToBDimYMap[CI], LoopIDToNumIterationsMap);
      /* insertCodeToAddWSS(Location, Allocation, wss); */
      auto aid_printer = insertConstantNode(Location, unsigned(AID->first));
      insertCodeToPrintGenericInt32(Location, aid_printer);
      insertCodeToAddWSS(Location, Allocation, wss_advanced, aid_printer);
      auto InvocationId = KernelInvocationToInvocationIDMap[CI];
      insertCodeToAddAIDToInvocationID(Location, AID->first, InvocationId);
      // for reuse
      if(FirstInvocation) {
          insertCodeToRecordReuse(FirstInvocation, InvocationId, AID->first, ExecutionCount, Allocation);
      }
      if(FirstInvocationNonIter ) {
          insertCodeToRecordReuse(FirstInvocationNonIter, InvocationId, AID->first, ExecutionCount, Allocation);
      }
    }
    // iterate over each allocation, and print the access count
    // TODO: Ensure that MalloPointerKernArgs contains only the exact pointers
    // that are passed to the kernel.
    for (auto Pointer = MallocPointerKernArgs.begin();
         Pointer != MallocPointerKernArgs.end(); Pointer++) {
      (*Pointer)->dump();
      // llvm::Value *AC = insertCodeToGetAccessCount(CI, *Pointer);
      // llvm::Value *Size = insertCodeToGetMemorySize(CI, *Pointer);
      // llvm::Value *AD = insertCodeToGetAccessDensity(CI, *Pointer);
    }

    auto InvocationInsertionPoint = KernelInvocationToInsertionPointMap[CI];
    insertCodeToPerformInvocationMemoryMgmt(InvocationInsertionPoint, CI);

    return;
  }

  Value *estimateWorkingSetSize(Instruction *Location, Value *Pointer,
                                Value *PD_bidx, Value *PD_bidy, Value *PD_phi,
                                Value *LoopIters, Value *BDimx, Value *BDimy,
                                Value *GDimx, Value *GDimy) {
    // auto pd_bidx = getPartDiff_bidx(Location, Pointer);
    // auto pd_bidy = getPartDiff_bidy(Location, Pointer);
    // auto pd_phi = getPartDiff_phi(Location, Pointer);
    errs() << "estimating working set size\n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(Pointer, Builder.getInt64Ty());
    Value *Args[] = {PD_bidx, PD_bidy, PD_phi, LoopIters,
                     BDimx,   BDimy,   GDimx,  GDimy};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee estimateWorkingSet =
        F->getParent()->getOrInsertFunction(
            "estimate_working_set", Type::getInt64Ty(Ctx),
            Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx),
            Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx),
            Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
    llvm::CallInst *wss = Builder.CreateCall(estimateWorkingSet, Args);
    insertCodeToPrintGenericInt64(Location, wss);
    return wss;
  }

  llvm::Value *getPartDiff_bidx(Instruction *Location, Value *P) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee GetPartDiff_bidx = F->getParent()->getOrInsertFunction(
        "get_pd_bidx", Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx));
    llvm::CallInst *PartDiff_bidx = Builder.CreateCall(GetPartDiff_bidx, Args);
    insertCodeToPrintGenericInt64(Location, PartDiff_bidx);
    return PartDiff_bidx;
  }

  llvm::Value *getPartDiff_bidy(Instruction *Location, Value *P) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee GetPartDiff_bidy = F->getParent()->getOrInsertFunction(
        "get_pd_bidy", Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx));
    llvm::CallInst *PartDiff_bidy = Builder.CreateCall(GetPartDiff_bidy, Args);
    insertCodeToPrintGenericInt64(Location, PartDiff_bidy);
    return PartDiff_bidy;
  }

  llvm::Value *getPartDiff_phi(Instruction *Location, Value *P) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    /* V->getType()->dump(); */
    llvm::Value *Ptr = Builder.CreatePtrToInt(P, Builder.getInt64Ty());
    Value *Args[] = {Ptr};
    // Builder.CreateCall(Fn, Args);
    llvm::FunctionCallee GetPartDiff_phi = F->getParent()->getOrInsertFunction(
        "get_pd_phi", Type::getInt64Ty(Ctx), Type::getInt64Ty(Ctx));
    llvm::CallInst *PartDiff_phi = Builder.CreateCall(GetPartDiff_phi, Args);
    insertCodeToPrintGenericInt64(Location, PartDiff_phi);
    return PartDiff_phi;
  }

  void identifyUnknownsFromExpressionTreeAdvanced(
          Instruction *Location, CallBase *CI,
          std::map<ExprTreeNodeAdvanced *, Value *> &Unknowns, ExprTreeNodeAdvanced *node) {
      /* std::vector<Value*> unknowns; */
      if (node == nullptr)
          return;
      // errs() << "unknown check? " << node->original_str << "\n";
      if (isTerminal(node)) {
          /* errs() << "unknown check? is terminal\n" << "\n"; */
          if (node->op == ETO_ARG) {
              errs() << "unknown: arg " << node->arg << " \n";
              auto unknown = KernelInvocationToArgNumberToActualArgMap[CI][node->arg];
              unknown->dump();
              Unknowns[node] = (unknown);
              return;
          }
          if (node->op == ETO_BDIMX) {
              /* errs() << "unknown: bdimx " << node->arg << " \n"; */
              unsigned bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
              /* unknown->dump(); */
              auto unknown = insertConstantNode(Location, bdimx);
              Unknowns[node] = (unknown);
              return;
          }
          if (node->op == ETO_BDIMY) {
              /* errs() << "unknown: bdimx " << node->arg << " \n"; */
              unsigned bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
              /* unknown->dump(); */
              auto unknown = insertConstantNode(Location, bdimy);
              Unknowns[node] = (unknown);
              return;
          }
          // Assertion will only work if all cases are handled.
          // assert(false);
      } else {
          /* errs() << "unknown check? not terminal\n" << "\n"; */
          for(auto child = node->children.begin(); child != node->children.end(); child++) {
              identifyUnknownsFromExpressionTreeAdvanced(Location, CI, Unknowns, *child);
          }
      }
  }

  void identifyUnknownsFromExpressionTree(
          Instruction *Location, CallBase *CI,
          std::map<ExprTreeNode *, Value *> &Unknowns, ExprTreeNode *node) {
      /* std::vector<Value*> unknowns; */
      if (node == nullptr)
          return;
      // errs() << "unknown check? " << node->original_str << "\n";
      if (isTerminal(node)) {
          /* errs() << "unknown check? is terminal\n" << "\n"; */
          if (node->op == ETO_ARG) {
              /* errs() << "unknown: arg " << node->arg << " \n"; */
              auto unknown = KernelInvocationToArgNumberToActualArgMap[CI][node->arg];
              /* unknown->dump(); */
              Unknowns[node] = (unknown);
              return;
          }
          if (node->op == ETO_BDIMX) {
              /* errs() << "unknown: bdimx " << node->arg << " \n"; */
              unsigned bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
              /* unknown->dump(); */
              auto unknown = insertConstantNode(Location, bdimx);
              Unknowns[node] = (unknown);
              return;
          }
          if (node->op == ETO_BDIMY) {
              /* errs() << "unknown: bdimx " << node->arg << " \n"; */
              unsigned bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
              /* unknown->dump(); */
              auto unknown = insertConstantNode(Location, bdimy);
              Unknowns[node] = (unknown);
              return;
          }
          // Assertion will only work if all cases are handled.
          // assert(false);
      } else {
          /* errs() << "unknown check? not terminal\n" << "\n"; */
          identifyUnknownsFromExpressionTree(Location, CI, Unknowns,
                  node->children[0]);
          identifyUnknownsFromExpressionTree(Location, CI, Unknowns,
                  node->children[1]);
      }
  }

  Value *castToDouble(Instruction *Location, Value *val) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    if(val->getType()->isIntegerTy(32)) {
      llvm::Value *Dst = Builder.CreateUIToFP(val, Type::getDoubleTy(Ctx));
      return Dst;
    }
    if(val->getType()->isDoubleTy()) {
      return val;
    }
    return nullptr;
  }

  Value *insertComparisonNode(Instruction *Location, Value *Src1, Value *Src2,
                               ExprTreeOp Op) {
    assert(Src1 != nullptr);
    assert(Src2 != nullptr);
    assert(Src1->getType() == Src2->getType());
    errs() << "insert comparions node\n";
    // insert conversion to double
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    Value *C1 = castToDouble(Location, Src1);
    Value *C2 = castToDouble(Location, Src2);
    llvm::Value *Dst = Builder.CreateFDiv(C2, C1);
    Dst->getType()->dump();
    return Dst;
    /* return nullptr; */
  }

  Value *insertComputationNode(Instruction *Location, Value *Src1, Value *Src2,
                               ExprTreeOp Op) {
    assert(Src1 != nullptr);
    assert(Src2 != nullptr);
      // cast to compatible types
    if(Src1->getType() != Src2->getType()) {
        if(Src1->getType()->isIntegerTy() && Src2->getType()->isIntegerTy()) {
            if(Src1->getType()->isIntegerTy(32)) {
                Src1 = insertCodeToCastInt32ToInt64(Location, Src1);
            }
            if(Src2->getType()->isIntegerTy(32)) {
                Src2 = insertCodeToCastInt32ToInt64(Location, Src2);
            }
        }
    }
    assert(Src1->getType() == Src2->getType());
    errs() << "insert computation node\n";
    errs() << Op << "\n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    if (Op == ETO_ADD) {
      llvm::Value *Dst = Builder.CreateAdd(Src1, Src2);
      // errs() << "insert computation node\n";
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_SUB) {
      llvm::Value *Dst = Builder.CreateSub(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_AND) {
      llvm::Value *Dst = Builder.CreateAnd(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_OR) {
      llvm::Value *Dst = Builder.CreateOr(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_MUL) {
      llvm::Value *Dst = Builder.CreateMul(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_SHL) {
      llvm::Value *Dst = Builder.CreateShl(Src2, Src1); // thanks to our convention. TODO: fix convention
      /* llvm::Value *Dst = Builder.CreateShl(Src1, Src2); // thanks to our convention. TODO: fix convention */
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_DIV) {
      llvm::Value *Dst = Builder.CreateUDiv(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_UDIV) {
      llvm::Value *Dst = Builder.CreateUDiv(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_SDIV) {
      llvm::Value *Dst = Builder.CreateSDiv(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    assert("shoudl not reachhere\n");
    return nullptr;
  }

  Value *insertComputationNodeAdvanced(Instruction *Location, Value *Src1, Value *Src2,
                               ExprTreeOp Op) {
    assert(Src1 != nullptr);
    assert(Src2 != nullptr);
      // cast to compatible types
    if(Src1->getType() != Src2->getType()) {
        if(Src1->getType()->isIntegerTy() && Src2->getType()->isIntegerTy()) {
            if(Src1->getType()->isIntegerTy(32)) {
                Src1 = insertCodeToCastInt32ToInt64(Location, Src1);
            }
            if(Src2->getType()->isIntegerTy(32)) {
                Src2 = insertCodeToCastInt32ToInt64(Location, Src2);
            }
        }
    }
    assert(Src1->getType() == Src2->getType());
    errs() << "insert computation node\n";
    errs() << Op << "\n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    if (Op == ETO_ADD) {
      llvm::Value *Dst = Builder.CreateAdd(Src1, Src2);
      // errs() << "insert computation node\n";
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_SUB) {
      llvm::Value *Dst = Builder.CreateSub(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_AND) {
      llvm::Value *Dst = Builder.CreateAnd(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_OR) {
      llvm::Value *Dst = Builder.CreateOr(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_MUL) {
      llvm::Value *Dst = Builder.CreateMul(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_SHL) {
      /* llvm::Value *Dst = Builder.CreateShl(Src2, Src1); // thanks to our convention. TODO: fix convention */
      llvm::Value *Dst = Builder.CreateShl(Src1, Src2); // thanks to our convention. TODO: fix convention
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_DIV) {
      llvm::Value *Dst = Builder.CreateUDiv(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_UDIV) {
      llvm::Value *Dst = Builder.CreateUDiv(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    if (Op == ETO_SDIV) {
      llvm::Value *Dst = Builder.CreateSDiv(Src1, Src2);
      Dst->getType()->dump();
      return Dst;
    }
    assert("shoudl not reachhere\n");
    return nullptr;
  }

  // This fuction is used to insert a constant node in the LLVM IR
  Value *insertConstantNode(Instruction *Location, ExprTreeNode *Node) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    llvm::Value *Dst = Builder.getInt32(Node->value);
    return Dst;
  }

  // This fuction is used to insert a constant node in the LLVM IR
  Value *insertConstantNode(Instruction *Location, ExprTreeNodeAdvanced *Node) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    llvm::Value *Dst = Builder.getInt32(Node->value);
    return Dst;
  }

  // This fuction is used to insert a constant node in the LLVM IR
  // overloaded for fun and (no) profit. will cause pain. someday.
  Value *insertConstantNode(Instruction *Location, unsigned value) {
      errs() << "insert constant node " << value << " \n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    llvm::Value *Dst = Builder.getInt32(value);
    return Dst;
  }

  Value *insertConstantNode(Instruction *Location, int value) {
      errs() << "insert constant node " << value << " \n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    llvm::Value *Dst = Builder.getInt32(value);
    return Dst;
  }

  // This fuction is used to insert a constant node in the LLVM IR
  // overloaded for fun and (no) profit. will cause pain. someday.
  Value *insertConstantNode(Instruction *Location, unsigned long long value) {
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    llvm::Value *Dst = Builder.getInt64(value);
    return Dst;
  }

  // This fuction is used to insert a constant node in the LLVM IR
  // overloaded for fun and (no) profit. will cause pain. someday.
  Value *insertConstantNode(Instruction *Location, bool value) {
      errs() << "insert constant node " << value << " \n";
    Function *F = Location->getParent()->getParent();
    LLVMContext &Ctx = F->getContext();
    IRBuilder<> Builder(Location);
    Builder.SetInsertPoint(Location);
    
    llvm::Value *Dst;
    if(value) {
        Dst = Builder.getTrue();
    } else {
        Dst = Builder.getTrue();
    }
    return Dst;
  }

  // TODO: Unknowns is not the correct word for describing what is currently
  // called so.
  Value * insertLoopItersEvaluationCode(Instruction *Location, CallBase *CI,
                                std::map<ExprTreeNode *, Value *> Unknowns,
                                ExprTreeNode *node) {
    if (node == nullptr)
      return nullptr;
    if (isTerminal(node)) {
      // handle this node
      errs() << "iliec: " << node->original_str << "\n";
      if (Unknowns.find(node) != Unknowns.end()) {
        Value *val = Unknowns[node];
        val->dump();
        return val;
      }
      if (node->op == ETO_CONST) { // currently assuming all constants are 32
                                   // bit unsigned integers
        errs() << "node value = " << node->value << "\n";
        node->value =
            stoi(node->original_str); // TODO: remove this hack: ensure that
                                      // the value is set correctly at some
                                      // earlier point
        return insertConstantNode(Location, node);
      }
      if (node->op == ETO_BIDX) { // fix with co-efficient based solution
        return insertConstantNode(Location, unsigned(0));
      }
      if (node->op == ETO_BIDY) { // fix with co-efficient based solution
        return insertConstantNode(Location, unsigned(0));
      }
      if (node->op == ETO_TIDX) { // fix with co-efficient based solution
        return insertConstantNode(Location, unsigned(0));
      }
      if (node->op == ETO_TIDY) { // fix with co-efficient based solution
        return insertConstantNode(Location, unsigned(0));
      }
      if (node->op == ETO_INCOMP) { // fix with co-efficient based solution
        return insertConstantNode(Location, unsigned(0));
      }
      assert(false); // must not reach here.
    } else {
      // handle phi nodes
      llvm::Value *Left = insertLoopItersEvaluationCode(Location, CI, Unknowns,
                                                        node->children[0]);
      llvm::Value *Right = insertLoopItersEvaluationCode(Location, CI, Unknowns,
                                                         node->children[1]);
      // handle this node
      // create a new LLVM value (Instruction) with the operation in question
      // it will take the left and right children as operands
      // return the new LLVM value
      return insertComputationNode(Location, Left, Right, node->op);
    }
    return nullptr;
  }

  Value * insertIfProbEvalCode(Instruction *Location, CallBase *CI,
                                std::map<ExprTreeNode *, Value *> Unknowns,
                                ExprTreeNode *node) {
    if (node == nullptr)
      return nullptr;
    if (isTerminal(node)) {
      // handle this node
      errs() << "iliec: " << node->original_str << "\n";
      if (Unknowns.find(node) != Unknowns.end()) {
        Value *val = Unknowns[node];
        val->dump();
        return val;
      }
      if (node->op == ETO_CONST) { // currently assuming all constants are 32
                                   // bit unsigned integers
        errs() << "node value = " << node->value << "\n";
        node->value =
          stoi(node->original_str); // TODO: remove this hack: ensure that
                                    // the value is set correctly at some
                                    // earlier point
        return insertConstantNode(Location, node);
      }
      if (node->op == ETO_BIDY) { // fix with co-efficient based solution
        errs() << "bidy \n";
        identifyMaxForUnknows(Location, CI, Unknowns, node);
        Value *maxValue = insertCodeToEstimateMaxValue(Location, CI, node, Unknowns,
            0, 0);
        auto one = insertConstantNode(Location, (unsigned)1);
        return insertComputationNode(Location, maxValue, one, ETO_SUB);
      }
      if (node->op == ETO_BIDX) { // fix with co-efficient based solution
        errs() << "bidx \n";
        identifyMaxForUnknows(Location, CI, Unknowns, node);
        Value *maxValue = insertCodeToEstimateMaxValue(Location, CI, node, Unknowns,
            0, 0);
        auto one = insertConstantNode(Location, (unsigned)1);
        return insertComputationNode(Location, maxValue, one, ETO_SUB);
      }
      if (node->op == ETO_TIDX) { // fix with co-efficient based solution
        errs() << "tidx \n";
        identifyMaxForUnknows(Location, CI, Unknowns, node);
        Value *maxValue = insertCodeToEstimateMaxValue(Location, CI, node, Unknowns,
            0, 0);
        auto one = insertConstantNode(Location, (unsigned)1);
        return insertComputationNode(Location, maxValue, one, ETO_SUB);
      }
      if (node->op == ETO_TIDY) { // fix with co-efficient based solution
        errs() << "tidy \n";
        identifyMaxForUnknows(Location, CI, Unknowns, node);
        Value *maxValue = insertCodeToEstimateMaxValue(Location, CI, node, Unknowns,
            0, 0);
        auto one = insertConstantNode(Location, (unsigned)1);
        return insertComputationNode(Location, maxValue, one, ETO_SUB);
      }
      assert(false); // must not reach here.
    } else {
      // handle phi nodes
      llvm::Value *Left = insertIfProbEvalCode(Location, CI, Unknowns,
          node->children[0]);
      llvm::Value *Right = insertIfProbEvalCode(Location, CI, Unknowns,
          node->children[1]);
      // handle this node
      // create a new LLVM value (Instruction) with the operation in question
      // it will take the left and right children as operands
      // return the new LLVM value
      if(node->op == ETO_ICMP || node->children[0]->isProb || node->children[1]->isProb) {
        node->isProb = true;
        errs() << node->op << "\n";
        return insertComparisonNode(Location, Left, Right, node->op);
      } else {
        errs() << node->op << "\n";
        return insertComputationNode(Location, Left, Right, node->op);
      }
    }
    return nullptr;
  }

  ExprTreeNode *partiallyEvaluatedLoopIters(Instruction *Location, CallBase *CI,
                                            std::string kernelName,
                                            int loopID) {
    if (loopID == 0)
      return nullptr;
    std::map<unsigned, std::vector<std::string>> kernelLoopToBoundsMap =
        LoopIDToLoopBoundsMap[kernelName];
    if (kernelLoopToBoundsMap.find(loopID) != kernelLoopToBoundsMap.end()) {
      // compute the loop bound after substituting actual values
      std::vector<std::string> LoopBoundsTokens = kernelLoopToBoundsMap[loopID];
      /* errs() << "\nloop tokens\n"; */
      ExprTreeNode *In, *Fin, *Step;
      unsigned long long InVal, FinVal, StepVal;
      std::vector<std::string> SplitTokens[3];
      unsigned currentSplit = 0;
      for (auto Token = LoopBoundsTokens.begin();
           Token != LoopBoundsTokens.end(); Token++) {
        errs() << *Token << " ";
        if ((*Token).compare("IN") == 0) {
          currentSplit = 0;
        } else if ((*Token).compare("FIN") == 0) {
          currentSplit = 1;
        } else if ((*Token).compare("STEP") == 0) {
          currentSplit = 2;
        } else {
          SplitTokens[currentSplit].push_back(*Token);
        }
      }
      In = createExpressionTree(SplitTokens[0]);
      Fin = createExpressionTree(SplitTokens[1]);
      Step = createExpressionTree(SplitTokens[2]);
      if(In == nullptr || Fin == nullptr) {
          return nullptr;
      }
      // TODO: simplify, combine and return
      ExprTreeNode *FinMinusIn = doOperationOnNodes(ETO_SUB, Fin, In);
      ExprTreeNode *FinMinusInDivStep =
          doOperationOnNodes(ETO_DIV, FinMinusIn, Step);
      return FinMinusInDivStep;
    }
    return nullptr;
  }

  ExprTreeNode *doOperationOnNodes(ExprTreeOp Op, ExprTreeNode *Left,
                                   ExprTreeNode *Right) {
    ExprTreeNode *Result = new ExprTreeNode();
    Result->op = Op;
    Result->children[0] = Left;
    Result->children[1] = Right;
    return Result;
  }

  bool isPointerChase(ExprTreeNode *root) {
    if (root->op == ETO_PC) {
      return true;
    }
    return false;
  }

  bool isIndirectAccess(ExprTreeNodeAdvanced* root) {
      if (root == nullptr)
          return false;
      std::stack<ExprTreeNodeAdvanced *> Stack;
      Stack.push(root);
      bool found_load = false;
      while (!Stack.empty()) {
          ExprTreeNodeAdvanced *Current = Stack.top();
          /* errs() << Current->original_str << " "; */
          if(root->op == ETO_LOAD) {
              if (found_load == true) {
                  return true;
              } else {
                  found_load = true;
              }
          }
          Stack.pop();
          /* if (isOperation(Current)) { */
              for(auto child = Current->children.begin(); child != Current->children.end(); child++) {
                  Stack.push(*child);
              }
          /* } */
      }
      return false;
  }

  void processKernelInvocation(CallBase *CI) {
    /* errs() << "enclosing function\n"; */
    KernelInvocationToEnclosingFunction[CI] = CI->getParent()->getParent();
    /* CI->getParent()->getParent()->dump(); */
  }

  // identify if a kernel invocation is inside a loop or not
  bool identifyIterative(CallBase *CI, LoopInfo &LI, ScalarEvolution &SE) {
    Loop *loop;
    if (loop = LI.getLoopFor(CI->getParent())) {
      errs() << "loop found\n";
      loop->dump();
      KernelInvocationToEnclosingLoopPredMap[CI] =
          loop->getLoopPredecessor()->getFirstNonPHI();
      auto *LIV = (loop)->getInductionVariable(SE);
      if (LIV) {
        errs() << "LIV : ";
        LIV->dump();
        KernelInvocationToEnclosingLIVMap[CI] = LIV;
      }
      auto CLIV = loop->getCanonicalInductionVariable();
      if (CLIV) {
        errs() << "CLIV : ";
        CLIV->dump();
      } else {
        errs() << "LIV not found\n";
        llvm::BasicBlock *Loopheader = loop->getHeader();
        for (auto &I : *Loopheader) {
          if (auto *PN = dyn_cast<PHINode>(&I)) {
            errs() << "PHI node found\n";
            PN->dump();
            // get scev node for the phinode
            // KernelInvocationToEnclosingLoopMap[CI] = PN;
          }
        }
      }
      auto loopbounds = loop->getBounds(SE);
      if (loopbounds) {
        Value &VInitial = loopbounds->getInitialIVValue();
        VInitial.dump();
        auto VI = getExpressionTree(&VInitial);
        errs() << "VI = " << evaluateRPNForIter0(CI, VI);
        auto VIC = evaluateRPNForIter0(CI, VI);
        errs() << "VI = " << VIC << "\n";
        Value &VFinal = loopbounds->getFinalIVValue();
        VFinal.dump();
        auto VF = getExpressionTree(&VFinal);
        auto VFC = evaluateRPNForIter0(CI, VF);
        errs() << "VF = " << VFC << "\n";
        Value *VSteps = loopbounds->getStepValue();
        VSteps->dump();
        auto VS = getExpressionTree(VSteps);
        auto VSC = evaluateRPNForIter0(CI, VS);
        errs() << "VS = " << VSC << "\n";
        KernelInvocationToIterMap[CI] = (VFC - VIC) / VSC;
        KernelInvocationToStepsMap[CI] = VSC;
      } else {
        errs() << "bound not found\n";
      }
      return true;
    }
    return false;
  }

  void findAndAddLocalFunction(Module &M) {
    for (auto &F : M) {
      if (F.isDeclaration()) {
        continue;
        ;
      }
      if (F.getName().contains("stub")) {
        errs() << "not running on " << F.getName() << "\n";
        continue;
      }
      // errs() << "locally defined function : " << F.getName() << "\n";
      ListOfLocallyDefinedFunctions.insert(&F);
    }
    return;
  }

  void extractArgsFromFunctionDefinition(Function &F) {
    if (F.isDeclaration()) {
      return;
    }
    // errs() << F.getName().str() << "\n";
    for (auto &Arg : F.args()) {
      // Arg.dump();
      FunctionToFormalArgumentMap[&F].push_back(&Arg);
      TerminalValues.insert(&Arg);
    }
    return;
  }

  void extractArgsFromFunctionCallSites(CallBase *CI) {
    // CI->dump();
    if (CI->getCalledFunction() == nullptr) {
      errs() << "FUNCTION CALL is probably indirect\n";
      return;
    }
    errs() << "CALL TO " << CI->getCalledFunction()->getName().str() << "\n";
    for (auto &Arg : CI->args()) {
      // Arg->dump();
      FunctionCallToActualArumentsMap[CI].push_back(Arg);
    }
  }

  void mapFormalArgumentsToActualArguments() {
    errs() << "MAPPING FORMAL ARGUMENTS TO ACTUAL ARGUMENTS\n\n";
    for (auto FnIter = FunctionToFormalArgumentMap.begin();
         FnIter != FunctionToFormalArgumentMap.end(); FnIter++) {
      errs() << "Function Name: " << FnIter->first->getName() << "\n";
      auto MatchCount = 0;
      for (auto CallSiteIter = FunctionCallToActualArumentsMap.begin();
           CallSiteIter != FunctionCallToActualArumentsMap.end();
           CallSiteIter++) {
        errs() << "Call site: "
               << CallSiteIter->first->getCalledFunction()->getName() << "\n";
        if (CallSiteIter->first->getCalledFunction() == FnIter->first) {
          errs() << "MATCH!\n";
          MatchCount++;
          for (auto FormalArgIter = FnIter->second.begin();
               FormalArgIter != FnIter->second.end(); FormalArgIter++) {
            /* errs() << (*FormalArgIter) << "\n"; */
            /* (*FormalArgIter)->dump(); */
          }
          for (auto ActualArgIter = CallSiteIter->second.begin();
               ActualArgIter != CallSiteIter->second.end(); ActualArgIter++) {
            /* errs() << (*ActualArgIter) << "\n"; */
            /* (*ActualArgIter)->dump(); */
          }
          for (unsigned long i = 0; i < FnIter->second.size(); i++) {
            auto *FormalArg = FnIter->second[i];
            auto *ActualArg = CallSiteIter->second[i];
            FormalArgumentToActualArgumentMap[FormalArg].push_back(ActualArg);
            /* auto ActualArgumentToFormalArgumentMap = */
            FunctionCallToActualArgumentToFormalArgumentMap[CallSiteIter->first]
                                                           [ActualArg] =
                                                               FormalArg;
            /* ActualArgumentToFormalArgumentMap[ActualArg] = (FormalArg); */
            CallSiteIter->first->dump();
            errs() << "formal arg to actual arg\n";
            errs() << FormalArg << "\n";
            FormalArg->dump();
            errs() << ActualArg << "\n";
            ActualArg->dump();
            FunctionCallToFormalArgumentToActualArgumentMap[CallSiteIter->first]
                                                           [FormalArg] =
                                                               (ActualArg);
          }
        }
      }
      if (MatchCount > 1) {
        errs() << "MORE THAN ONE CALL SITE \n";
      }
    }
  }

  void analyzePointerPropogationRecursive(CallBase *CI) {
    if (VisitedCallInstForPointerPropogation.find(CI) !=
        VisitedCallInstForPointerPropogation.end()) {
      return;
    } else {
      VisitedCallInstForPointerPropogation.insert(CI);
    }
    auto *Func = CI->getCalledFunction();
    errs() << "function name = " << Func->getName() << "\n";
    /* Func->dump(); */
    if (ListOfLocallyDefinedFunctions.find(Func) ==
        ListOfLocallyDefinedFunctions.end()) {
      errs() << "not locally define\n";
      return;
    }
    for (auto &BB : *Func) {
      for (auto &I : BB) {
        if (isa<AllocaInst>(I)) {
          auto AI = dyn_cast<AllocaInst>(&I);
          I.dump();
          AI->getType()->dump();
          AI->getAllocatedType()->dump();
          if (I.getType()->isPointerTy()) {
            OriginalPointers.insert(&I);
            PointerOpToOriginalPointers[&I] = &I;
          }
          if (auto Stype = dyn_cast<StructType>(AI->getAllocatedType())) {
            errs() << "Struct Type\n";
            StructAllocas.insert(AI);
          }
        }
      }
    }
    for (auto &BB : *Func) {
      for (auto &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          // LI->getPointerOperand()->dump();
          auto POGP = PointerOpToOriginalPointers.find(LI->getPointerOperand());
          if (POGP != PointerOpToOriginalPointers.end()) {
            PointerOpToOriginalPointers[LI] = POGP->second;
            errs() << "\nLOAD INST \n";
            LI->dump();
            POGP->second->dump();
            if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
              PointerOpToOriginalStructPointer[LI] = POGP->second;
              PointerOpToOriginalStructPointersIndex[LI] =
                  PointerOpToOriginalStructPointersIndex
                      [LI->getPointerOperand()];
              errs() << "zoo zoo = "
                     << PointerOpToOriginalStructPointersIndex
                            [LI->getPointerOperand()];
            }
          }
        }
        if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
          // LI->getPointerOperand()->dump();
          errs() << "GEPI testing: ";
          GEPI->dump();
          GEPI->getPointerOperand()->dump();
          errs() << GEPI->getPointerOperand() << "\n";
          auto POGP =
              PointerOpToOriginalPointers.find(GEPI->getPointerOperand());
          if (POGP != PointerOpToOriginalPointers.end()) {
            PointerOpToOriginalPointers[GEPI] = POGP->second;
            errs() << "\nGEPI INST \n";
            GEPI->dump();
            GEPI->getPointerOperand()->dump();
            POGP->second->dump();
            if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
              PointerOpToOriginalStructPointer[GEPI] = POGP->second;
              auto numIndices = GEPI->getNumIndices();
              if (numIndices == 2) {
                if (auto FieldNum =
                        dyn_cast<ConstantInt>(GEPI->getOperand(2))) {
                  errs() << "og is struct\n";
                  PointerOpToOriginalStructPointersIndex[GEPI] =
                      FieldNum->getSExtValue();
                  errs() << "field num = " << FieldNum << "\n";
                }
              } else {
                if (auto FieldNum =
                        dyn_cast<ConstantInt>(GEPI->getOperand(1))) {
                  errs() << "og maybe struct or array\n";
                  PointerOpToOriginalStructPointersIndex[GEPI] =
                      FieldNum->getSExtValue();
                  errs() << "field num = " << FieldNum << "\n";
                }
              }
            }
          }
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          // LI->getPointerOperand()->dump();
          auto POGP = PointerOpToOriginalPointers.find(SI->getValueOperand());
          if (POGP != PointerOpToOriginalPointers.end()) {
            PointerOpToOriginalPointers[SI->getPointerOperand()] = POGP->second;
            errs() << "\nSTORE INST \n";
            SI->dump();
            SI->getPointerOperand()->dump();
            POGP->second->dump();
            if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
              PointerOpToOriginalStructPointer[SI->getPointerOperand()] =
                  POGP->second;
              PointerOpToOriginalStructPointersIndex[SI->getPointerOperand()] =
                  PointerOpToOriginalStructPointersIndex[SI->getValueOperand()];
              errs() << "zoo zoo = "
                     << PointerOpToOriginalStructPointersIndex
                            [SI->getValueOperand()];
            }
          }
        }
        if (auto *CI = dyn_cast<CallBase>(&I)) {
          auto *Callee = CI->getCalledFunction();
          if ((Callee && ((Callee->getName() == "llvm.lifetime.start.p0") ||
                          Callee->getName() == "llvm.lifetime.end.p0"))) {
            continue;
          }
          if ((Callee && (Callee->getName() == "llvm.memcpy.p0.p0.i64"))) {
            errs() << "memcpy found\n";
            CI->getOperand(0)->dump();
            CI->getOperand(1)->dump();
            bool isStackVar0 = isa<AllocaInst>(CI->getOperand(0));
            bool isStackVar1 = isa<AllocaInst>(CI->getOperand(1));
            if (isStackVar0 || isStackVar1) {
              MemcpyOpForStructs.insert(CI);
              MemcpyOpForStructsSrcToInstMap[CI->getOperand(1)] = CI;
              MemcpyOpForStructsDstToInstMap[CI->getOperand(0)] = CI;
              if (PointerOpToOriginalPointers.find(CI->getOperand(1)) !=
                  PointerOpToOriginalPointers.end()) {
                errs() << "memcpy taint propogated\n";
                PointerOpToOriginalPointers[CI->getOperand(1)]->dump();
                PointerOpToOriginalPointers[CI->getOperand(0)] =
                    PointerOpToOriginalPointers[CI->getOperand(1)];
                auto OriginalPointer =
                    PointerOpToOriginalPointers[CI->getOperand(1)];
                if (StructAllocas.find(OriginalPointer) !=
                    StructAllocas.end()) {
                  PointerOpToOriginalStructPointer[CI->getOperand(0)] =
                      OriginalPointer;
                  PointerOpToOriginalStructPointersIndex[CI->getOperand(0)] =
                      PointerOpToOriginalStructPointersIndex[CI->getOperand(1)];
                  errs()
                      << "zoo zoo = "
                      << PointerOpToOriginalStructPointersIndex[CI->getOperand(
                             1)];
                }
              }
            }
          }
          errs() << "CallBase : ";
          CI->dump();
          auto *Func = CI->getCalledFunction();
          if (ListOfLocallyDefinedFunctions.find(Func) ==
              ListOfLocallyDefinedFunctions.end()) {
            continue;
          }
          /* Func->dump(); */
          for (auto &Arg : CI->args()) {
            auto POGP = PointerOpToOriginalPointers.find(Arg);
            if (POGP != PointerOpToOriginalPointers.end()) {
              errs() << "\nCALL INST \n";
              CI->dump();
              Arg->dump();
              auto *OriginalPointer = POGP->second;
              OriginalPointer->dump();
              auto ActualArgumentToFormalArgumentMap =
                  FunctionCallToActualArgumentToFormalArgumentMap[CI];
              auto *FormalArg =
                  ActualArgumentToFormalArgumentMap[Arg]; // ->dump();
              FormalArg->dump();
              errs() << FormalArg << "\n";
              PointerOpToOriginalPointers[FormalArg] = OriginalPointer;
              if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
                PointerOpToOriginalStructPointer[FormalArg] = POGP->second;
                PointerOpToOriginalStructPointersIndex[FormalArg] =
                    PointerOpToOriginalStructPointersIndex[Arg];
                errs() << "zoo zoo = "
                       << PointerOpToOriginalStructPointersIndex[Arg];
              }
            }
          }
          errs() << "Recurse into called functions\n";
          analyzePointerPropogationRecursive(CI);
        }
      }
    }
    return;
  }

  void analyzePointerPropogation(Module &M) {
    errs() << "POINTER COLLECTION IN MAIN\n";
    // PointerOpToOriginalPointers;
    // OriginalPointers;
    for (auto &F : M) {
      if (F.getName() != "main") {
        continue;
      }
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (isa<AllocaInst>(I)) {
            auto AI = dyn_cast<AllocaInst>(&I);
            I.dump();
            AI->getType()->dump();
            AI->getAllocatedType()->dump();
            if (I.getType()->isPointerTy()) {
              // OriginalPointers.insert(&I);
              PointerOpToOriginalPointers[&I] = &I;
            }
            if (auto Stype = dyn_cast<StructType>(AI->getAllocatedType())) {
              errs() << "Struct Type\n";
              StructAllocas.insert(AI);
            }
          }
        }
      }
    }
    errs() << "POINTER PROPOGATION\n";
    for (auto &F : M) {
      if (F.getName() != "main") {
        continue;
      }
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            // LI->getPointerOperand()->dump();
            auto POGP =
                PointerOpToOriginalPointers.find(LI->getPointerOperand());
            if (POGP != PointerOpToOriginalPointers.end()) {
              PointerOpToOriginalPointers[LI] = POGP->second;
              errs() << "\nLOAD INST \n";
              LI->dump();
              POGP->second->dump();
              if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
                PointerOpToOriginalStructPointer[LI] = POGP->second;
                PointerOpToOriginalStructPointersIndex[LI] =
                    PointerOpToOriginalStructPointersIndex
                        [LI->getPointerOperand()];
                errs() << "zoo zoo = "
                       << PointerOpToOriginalStructPointersIndex
                              [LI->getPointerOperand()];
              }
            }
          }
          if (auto *SI = dyn_cast<StoreInst>(&I)) {
            // LI->getPointerOperand()->dump();
            auto POGP = PointerOpToOriginalPointers.find(SI->getValueOperand());
            if (POGP != PointerOpToOriginalPointers.end()) {
              PointerOpToOriginalPointers[SI->getPointerOperand()] =
                  POGP->second;
              errs() << "\nSTORE INST \n";
              SI->dump();
              SI->getPointerOperand()->dump();
              POGP->second->dump();
              if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
                PointerOpToOriginalStructPointer[SI->getPointerOperand()] =
                    POGP->second;
                PointerOpToOriginalStructPointersIndex
                    [SI->getPointerOperand()] =
                        PointerOpToOriginalStructPointersIndex
                            [SI->getValueOperand()];
                errs() << "zoo zoo = "
                       << PointerOpToOriginalStructPointersIndex
                              [SI->getValueOperand()];
              }
            }
            if (isa<ConstantInt>(SI->getValueOperand())) {
              errs() << "Constant store\n";
              auto con = dyn_cast<ConstantInt>(SI->getValueOperand());
              PointerOpToOriginalConstant[SI->getPointerOperand()] =
                  con->getSExtValue();
            }
          }
          if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
            // LI->getPointerOperand()->dump();
            auto POGP =
                PointerOpToOriginalPointers.find(GEPI->getPointerOperand());
            if (POGP != PointerOpToOriginalPointers.end()) {
              PointerOpToOriginalPointers[GEPI] = POGP->second;
              errs() << "\nGEPI INST \n";
              GEPI->dump();
              GEPI->getPointerOperand()->dump();
              POGP->second->dump();
              if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
                PointerOpToOriginalStructPointer[GEPI] = POGP->second;
                auto numIndices = GEPI->getNumIndices();
                if (numIndices == 2) {
                  if (auto FieldNum =
                          dyn_cast<ConstantInt>(GEPI->getOperand(2))) {
                    errs() << "og is struct\n";
                    PointerOpToOriginalStructPointersIndex[GEPI] =
                        FieldNum->getSExtValue();
                    errs() << "field num = " << FieldNum << "\n";
                  }
                } else {
                  if (auto FieldNum =
                          dyn_cast<ConstantInt>(GEPI->getOperand(1))) {
                    errs() << "og maybe struct or array\n";
                    PointerOpToOriginalStructPointersIndex[GEPI] =
                        FieldNum->getSExtValue();
                    errs() << "field num = " << FieldNum << "\n";
                  }
                }
              }
            }
          }
          if (auto *CI = dyn_cast<CallBase>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if ((Callee && (Callee->getName() == "llvm.lifetime.start.p0")) ||
                (Callee && Callee->getName() == "llvm.lifetime.end.p0")) {
              continue;
            }
            if ((Callee && (Callee->getName() == "llvm.memcpy.p0.p0.i64"))) {
              errs() << "memcpy found\n";
              CI->getOperand(0)->dump();
              CI->getOperand(1)->dump();
              bool isStackVar0 = isa<AllocaInst>(CI->getOperand(0));
              bool isStackVar1 = isa<AllocaInst>(CI->getOperand(1));
              if (isStackVar0 || isStackVar1) {
                MemcpyOpForStructs.insert(CI);
                MemcpyOpForStructsSrcToInstMap[CI->getOperand(1)] = CI;
                MemcpyOpForStructsDstToInstMap[CI->getOperand(0)] = CI;
                if (PointerOpToOriginalPointers.find(CI->getOperand(1)) !=
                    PointerOpToOriginalPointers.end()) {
                  errs() << "memcpy taint propogated \n ";
                  PointerOpToOriginalPointers[CI->getOperand(1)]->dump();
                  PointerOpToOriginalPointers[CI->getOperand(0)] =
                      PointerOpToOriginalPointers[CI->getOperand(1)];
                  auto OriginalPointer =
                      PointerOpToOriginalPointers[CI->getOperand(1)];
                  if (StructAllocas.find(OriginalPointer) !=
                      StructAllocas.end()) {
                    PointerOpToOriginalStructPointer[CI->getOperand(0)] =
                        OriginalPointer;
                    PointerOpToOriginalStructPointersIndex[CI->getOperand(0)] =
                        PointerOpToOriginalStructPointersIndex[CI->getOperand(
                            1)];
                    errs() << "zoo zoo = "
                           << PointerOpToOriginalStructPointersIndex
                                  [CI->getOperand(1)];
                  }
                }
              }
            }
            if (ListOfLocallyDefinedFunctions.find(Callee) ==
                ListOfLocallyDefinedFunctions.end()) {
              continue;
            }
            for (auto &Arg : CI->args()) {
              auto POGP = PointerOpToOriginalPointers.find(Arg);
              if (POGP != PointerOpToOriginalPointers.end()) {
                errs() << "\nCALL INST \n";
                CI->dump();
                Arg->dump();
                errs() << Arg << "\n";
                auto *OriginalPointer = POGP->second;
                OriginalPointer->dump();
                auto ActualArgumentToFormalArgumentMap =
                    FunctionCallToActualArgumentToFormalArgumentMap[CI];
                Value *FormalArg =
                    ActualArgumentToFormalArgumentMap[Arg]; // ->dump();
                FormalArg->dump();
                errs() << FormalArg << "\n";
                PointerOpToOriginalPointers[FormalArg] = OriginalPointer;
                if (StructAllocas.find(POGP->second) != StructAllocas.end()) {
                  PointerOpToOriginalStructPointer[FormalArg] = POGP->second;
                  PointerOpToOriginalStructPointersIndex[FormalArg] =
                      PointerOpToOriginalStructPointersIndex[Arg];
                  errs() << "zoo zoo = "
                         << PointerOpToOriginalStructPointersIndex[Arg];
                }
              }
            }
            errs() << "Recurse into called functions\n";
            analyzePointerPropogationRecursive(CI);
          }
        }
      }
    }
    return;
  }

  void setTerminalsAndOperations() {
    terminals.insert(ETO_TIDX);
    terminals.insert(ETO_TIDY);
    terminals.insert(ETO_BIDX);
    terminals.insert(ETO_BIDY);
    terminals.insert(ETO_BDIMX);
    terminals.insert(ETO_BDIMY);
    terminals.insert(ETO_PHI_TERM);
    terminals.insert(ETO_ARG);
    terminals.insert(ETO_CONST);
    terminals.insert(ETO_INTERM);
    terminals.insert(ETO_INCOMP);

    operations.insert(ETO_ADD);
    operations.insert(ETO_AND);
    operations.insert(ETO_SUB);
    operations.insert(ETO_OR);
    operations.insert(ETO_MUL);
    operations.insert(ETO_UDIV);
    operations.insert(ETO_SDIV);
    operations.insert(ETO_SHL);
    operations.insert(ETO_PHI);
    operations.insert(ETO_ICMP);
  }

  void printLoopInformation() {
    errs() << "loop information\n";
    for (auto I = LoopIDToLoopItersMap.begin(); I != LoopIDToLoopItersMap.end();
         I++) {
      errs() << I->first << "\n";
      for (auto L = I->second.begin(); L != I->second.end(); L++) {
        errs() << L->first << " ==> " << L->second << "\n";
      }
    }
    for (auto I = LoopIDToLoopBoundsMap.begin();
         I != LoopIDToLoopBoundsMap.end(); I++) {
      errs() << I->first << "\n";
      for (auto L = I->second.begin(); L != I->second.end(); L++) {
        errs() << L->first << " ==> ";
        for (auto str = L->second.begin(); str != L->second.end(); str++) {
          errs() << *str << " ";
        }
      }
    }
    errs() << "\n";

    errs() << "phi to loop mapping\n";
    for (auto I = PhiNodeToLoopIDMap.begin();
         I != PhiNodeToLoopIDMap.end(); I++) {
        errs() << I->first << " " << I->second << "\n";
    }
    errs() << "\n";
  }

  void printAccessInformation() {
    errs() << "access information\n";
    errs() << "\n";
    // the key for all the AccessId maps is the same (kernel name), so use any
    // to iterate
    for (auto I = KernelNameToAccessIDToAllocationArgMap.begin();
         I != KernelNameToAccessIDToAllocationArgMap.end(); I++) {
      errs() << "\nkernel name: " << I->first << "\n";
      std::map<unsigned, unsigned> AccessIDToArgMap =
          KernelNameToAccessIDToAllocationArgMap[I->first];
      errs() << "AID to arg map\n";
      for (auto AID = AccessIDToArgMap.begin(); AID != AccessIDToArgMap.end();
           AID++) {
        errs() << AID->first << " " << AID->second << "\n";
      }
      errs() << "AID to loop map\n";
      std::map<unsigned, unsigned> AccessIDToLoopMap =
          KernelNameToAccessIDToEnclosingLoopMap[I->first];
      for (auto AID = AccessIDToLoopMap.begin(); AID != AccessIDToLoopMap.end();
           AID++) {
        errs() << AID->first << " " << AID->second << "\n";
      }
      errs() << "AID to expression tree map\n";
      std::map<unsigned, ExprTreeNode *> AccessIDToExprMap =
          KernelNameToAccessIDToExpressionTreeMap[I->first];
      std::map<unsigned, ExprTreeNodeAdvanced *> AccessIDToAdvancedExprMap =
          KernelNameToAccessIDToAdvancedExpressionTreeMap[I->first];
      for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end();
           AID++) {
        errs() << "\nAID = " << AID->first << "  ";
        traverseExpressionTree(AID->second);
      }
      for (auto AID = AccessIDToAdvancedExprMap.begin(); AID != AccessIDToAdvancedExprMap.end();
           AID++) {
        errs() << "\nAAID = " << AID->first << "  ";
        traverseExpressionTree(AID->second);
      }
    }
  }

  bool doInitialization(Module &M) override {
    setTerminalsAndOperations();
    printKernelDeviceAnalyis();

    printLoopInformation();
    printAccessInformation();

    // llvm::LLVMContext Ctx = M.getContext();

    FunctionType *FT = FunctionType::get(
        Type::getVoidTy(M.getContext()),
        {Type::getInt64Ty(M.getContext()), Type::getInt64Ty(M.getContext())},
        false);
    Function *Fn = Function::Create(FT, Function::ExternalLinkage,
                                    "addIntoAllocationMap", M);
    FunctionType *FT2 =
        FunctionType::get(Type::getVoidTy(M.getContext()), false);
    Function *Fn2 = Function::Create(FT, Function::ExternalLinkage,
                                     "printAllocationMap", M);

    return false;
  }

  bool runOnModule(Module &M) override {

    findAndAddLocalFunction(M);
    for (auto *Fn : ListOfLocallyDefinedFunctions) {
      errs() << "Locally defined function " << Fn->getName().str() << "\n";
    }

    for (auto &F : M) {
      extractArgsFromFunctionDefinition(F);
    }
    for (auto FTFA = FunctionToFormalArgumentMap.begin();
         FTFA != FunctionToFormalArgumentMap.end(); FTFA++) {
      errs() << "Function name = " << FTFA->first->getName().str() << "\n";
      for (auto Arg = FTFA->second.begin(); Arg != FTFA->second.end(); Arg++) {
        errs() << "Arg name = ";
        (*Arg)->dump();
        errs() << "\n";
      }
    }

    for (auto &F : M) {
      if (F.getName().contains("stub")) {
        errs() << "not running on " << F.getName() << "\n";
        continue;
      }
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallBase>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if ((Callee && (Callee->getName() == "llvm.lifetime.start.p0")) ||
                (Callee && (Callee->getName() == "llvm.lifetime.end.p0"))) {
              continue;
            }
            extractArgsFromFunctionCallSites(CI);
          }
          if (auto *CI = dyn_cast<InvokeInst>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if ((Callee && (Callee->getName() == "llvm.lifetime.start.p0")) ||
                (Callee && (Callee->getName() == "llvm.lifetime.end.p0"))) {
              continue;
            }
            extractArgsFromFunctionCallSites(CI);
          }
        }
      }
    }

    mapFormalArgumentsToActualArguments();
    errs() << "\n\n FORMAL ARG TO ACTUAL ARG MAP\n\n";
    for (auto FATAAM = FormalArgumentToActualArgumentMap.begin();
         FATAAM != FormalArgumentToActualArgumentMap.end(); FATAAM++) {
      errs() << "formal arg\n";
      FATAAM->first->dump();
      errs() << "actual args\n";
      for (auto ActualArgIter = FATAAM->second.begin();
           ActualArgIter != FATAAM->second.end(); ActualArgIter++) {
        /* errs() << (*ActualArgIter) << "\n"; */
        (*ActualArgIter)->dump();
      }
    }

    analyzePointerPropogation(M);
    errs() << "\nPOINTER PROPOGATION RESULTS\n";
    for (auto POGP = PointerOpToOriginalPointers.begin();
         POGP != PointerOpToOriginalPointers.end(); POGP++) {
      errs() << "\n";
      POGP->first->dump();
      POGP->second->dump();
    }

    for (auto &F : M) {
      if (F.getName().contains("stub")) {
        errs() << "not running on " << F.getName() << "\n";
        continue;
      }
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallBase>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == "cudaMallocManaged") {
              processMemoryAllocation(CI);
            }
            if (Callee && Callee->getName() == ("cudaLaunchKernel")) {
            }
          }
        }
      }
    }

    errs() << "\nMALLOC SIZE MAP\n";
    for (auto I = MallocSizeMap.begin(); I != MallocSizeMap.end(); I++) {
      I->first->dump();
      if (auto *CI = dyn_cast<CallBase>(I->first)) {
        CI->getOperand(0)->dump();
      }
      errs() << "Size  " << I->second << "\n";
    }

    errs() << "\nmemory size printing\n";
    // quick insert test for printing memory size
    for (auto I = MallocSizeMap.begin(); I != MallocSizeMap.end(); I++) {
      I->first->dump();
      if (auto *CI = dyn_cast<CallBase>(I->first)) {
        /* CI->getOperand(1)->dump(); */
        // insertCodeToPrintAddress(CI, CI->getOperand(0));
        // insertCodeToPrintSize(CI, CI->getOperand(1));
        insertCodeToRecordMalloc(CI, CI->getOperand(0), CI->getOperand(1));
      }
    }

    // Note: we are computing the block size earlier/seperately from the main
    // loop below because of the push pop and sroa shenanigans.
    // We can make it more elagant using the processKernelShapeArguments method
    // to only collect information about push pop and SROA and then process it
    // in another method, called from the main loop.
    for (auto &F : M) {
      if (F.getName().contains("stub")) {
        errs() << "not running on " << F.getName() << "\n";
        continue;
      }
      // comment this for mummer
      processKernelShapeArguments(F);
    };
    errs() << "KERNEL INVOCATION TO BLOCK SIZE MAP\n";
    for (auto Iter = KernelInvocationToBlockSizeMap.begin();
         Iter != KernelInvocationToBlockSizeMap.end(); Iter++) {
      (*Iter).first->dump();
      for (auto BDimIter = (*Iter).second.begin();
           BDimIter != (*Iter).second.end(); BDimIter++) {
        errs() << (*BDimIter).first << " ";
        errs() << (*BDimIter).second << "\n";
      }
    }

    for (auto &F : M) {
      if (F.getName().contains("__cuda_module_ctor") ||
          F.getName().contains("__cuda_register_globals")) {
        errs() << "CTOR FOUND\n";
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallBase>(&I)) {
              CI->dump();
              auto *Callee = CI->getCalledFunction();
              if (Callee && Callee->getName() == ("__cudaRegisterFunction")) {
                errs() << "Found a registration\n";
                errs() << Callee->getName() << "\n";
                /* CI->getArgOperand(1)->dump(); */
                if (llvm::Function *Fn =
                        dyn_cast<Function>(CI->getArgOperand(1))) {
                  errs() << " func name = " << Fn->getName() << "\n";
                  if (llvm::GlobalVariable *DvFn =
                          dyn_cast<GlobalVariable>(CI->getArgOperand(2))) {
                    errs() << " device side name = ";
                    auto DvFnStr =
                        dyn_cast<ConstantDataArray>(DvFn->getInitializer());
                    if (DvFnStr) {
                      errs() << DvFnStr->getAsCString() << "\n";
                      HostSideKernelNameToOriginalNameMap[std::string(
                          Fn->getName())] =
                          std::string(DvFnStr->getAsCString());
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        continue;
      }
    }

    bool isIterative = false;
    std::set<Function *> FunctionsWithKernelLaunches;

    for (auto &F : M) {

      if (F.getName().contains("stub")) {
        errs() << "not running on " << F.getName() << "\n";
        continue;
      }
      for (auto &BB : F) {

        LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
        ScalarEvolution &SE =
            getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallBase>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName() == ("cudaLaunchKernel")) {

              KernelInvocationToInvocationIDMap[CI] = KernelInvocationID++;
              bool Iterative = identifyIterative(CI, LI, SE);
              if (Iterative) {
                KernelLaunchIsIterative[CI] = true;
              } else {
                KernelLaunchIsIterative[CI] = false;
              }

              addCodeToAddInvocationID(CI, KernelInvocationToInvocationIDMap[CI]);

              auto *KernelPointer = CI->getArgOperand(0);
              auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
              auto KernelName = KernelFunction->getName();
              errs() << "Name of kernel = " << KernelName << "\n";
              if (KernelName.compare("_ZN8GpuBTree7kernels25__device_stub__"
                                     "init_btreeI13PoolAllocatorEEvPjT_") == 0)
                continue;
              if (KernelName.compare(
                      "_ZN8GpuBTree7kernels26__device_stub__insert_"
                      "keysIjjj13PoolAllocatorEEvPjPT_PT0_T1_T2_") == 0)
                continue;
              if (KernelName.compare(
                      "_Z32__device_stub__"
                      "mummergpuRCKernelP10MatchCoordPcPKiS3_ii") == 0)
                continue;
              if (KernelName.compare(
                      "_Z26__device_stub__"
                      "printKernelP9MatchInfoiP9AlignmentPcP12_PixelOfNodeP16_"
                      "PixelOfChildrenPKiS9_iiiii") == 0)
                continue;
              KernelLaunches.push_back(CI);

              processKernelInvocation(CI);
              processKernelSignature(CI);
              processKernelArguments(CI);

              FunctionsWithKernelLaunches.insert(&F);
              errs() << "insert into FunctionsWithKernelLaunches\n";
            }
          }
        }
      }
    }
    /* return true; */

    errs() << "the numeber of invocations is " << KernelLaunches.size();
    if(KernelLaunches.size() > 1) {
        multiKernel = true;
    }

    bool LoopSingleRunFunctionInserted = false;
    bool FirstInvocationFound = false;
    errs() << "Kernel Launches listed here\n";
          Instruction *InsertionPoint;
    // iterate over kernel launches and insert code
    for (auto KL = KernelLaunches.begin(); KL != KernelLaunches.end(); KL++) {
      (*KL)->dump();
      if (KernelLaunchIsIterative[*KL]) {
        errs() << "Iterative kernel\n";
        // numThread computation may be hoisted, only if numThreads is
        // iteation-independent
        auto CI = dyn_cast<CallBase>(*KL);
        // for this invocation; inserts LLVM value that will hold the number of times loop will execute
        std::map<unsigned, Value *> LoopIDToNumIterationsMap;
        std::map<unsigned, bool> LoopIDToIncompMap;
        std::map<unsigned, Value *> IfIDToProbMap;
        if (gridSizeIsIterationIndependent(CI)) {
          errs() << "grid size is iteration independent\n";
          // identify  the instruction just above the loop header
          // auto *LoopHeader = getEnclosingLoopPreheaderFirst(CI);
          // LoopHeader->dump();
          // auto InsertionPoint = LoopHeader;
          auto *LIV = getEnclosingLoopInductionVariable(CI);
          Instruction *IterationDecisionPoint;
          if(LoopSingleRunFunctionInserted == false){
              if (LIVTOInsertionPointMap.find(LIV) != LIVTOInsertionPointMap.end()) {
                  InsertionPoint = LIVTOInsertionPointMap[LIV];
              } else {
                  InsertionPoint = insertCodeForFirstIterationExecution(CI, LIV);
                  /* IterationDecisionPoint = insertCodeForIterationDecision(CI, LIV); */
                  LIVTOInsertionPointMap[LIV] = InsertionPoint;
              }
              InsertionPoint = insertCodeToPerformIterativeMemoryMgmt(InsertionPoint);
              /* InsertionPoint = insertCodeToPerformMemoryMgmtIteration(InsertionPoint, LIV); */
              /* InsertionPoint = insertPoint; */
                  FirstInvocation = InsertionPoint;
              LoopSingleRunFunctionInserted = true;
          }
          KernelInvocationToInsertionPointMap[CI] = CI;
          insertCodeToComputeKernelLoopIterationCount(InsertionPoint, CI,
                                                      LoopIDToNumIterationsMap, LoopIDToIncompMap);
          identifyIterationDependentAccesses(InsertionPoint, CI,
                                             LoopIDToNumIterationsMap);
          std::map<unsigned, Value *> LoopIDToNumIterationsMap;
          std::map<CallBase *, Value *> KernelInvocationToBDimXMap;
          std::map<CallBase *, Value *> KernelInvocationToBDimYMap;
          std::map<CallBase *, Value *> KernelInvocationToGDimXMap;
          std::map<CallBase *, Value *> KernelInvocationToGDimYMap;
          llvm::Value *NumThreadsInGrid = insertCodeToPrintNumThreads(InsertionPoint,
              CI, KernelInvocationToBDimXMap, KernelInvocationToBDimYMap,
              KernelInvocationToGDimXMap, KernelInvocationToGDimYMap);
          insertCodeToComputeKernelLoopIterationCount(InsertionPoint, CI,
                                                      LoopIDToNumIterationsMap, LoopIDToIncompMap);
          /* insertCodeToComputeConditionalExecutionProbability(InsertionPoint, CI, IfIDToProbMap); */
          // // NumThreadsInGrid and LoopIDToNumIterationsMap are sufficient to
          // // compute the total number of exectutions of any particular
          // // memory operation in the kernel
          insertCodeToComputeAccessDensity(InsertionPoint,
                  CI, NumThreadsInGrid, LoopIDToNumIterationsMap, LoopIDToIncompMap,
                  KernelInvocationToBDimXMap, KernelInvocationToBDimYMap,
                  KernelInvocationToGDimXMap, KernelInvocationToGDimYMap);
        } else {
          errs() << "grid size is not iteration independent\n";
          errs() << "we will assume that kernel is not iterative and perform other optimization\n";
        if (llvm::CallBase *CI = dyn_cast<CallBase>(*KL)) {
          std::map<unsigned, Value *> LoopIDToNumIterationsMap;
        std::map<unsigned, bool> LoopIDToIncompMap;
        std::map<unsigned, Value *> IfIDToProbMap;
          std::map<CallBase *, Value *> KernelInvocationToBDimXMap;
          std::map<CallBase *, Value *> KernelInvocationToBDimYMap;
          std::map<CallBase *, Value *> KernelInvocationToGDimXMap;
          std::map<CallBase *, Value *> KernelInvocationToGDimYMap;
          llvm::Value *NumThreadsInGrid = insertCodeToPrintNumThreads(CI,
              CI, KernelInvocationToBDimXMap, KernelInvocationToBDimYMap,
              KernelInvocationToGDimXMap, KernelInvocationToGDimYMap);
          insertCodeToComputeKernelLoopIterationCount(CI, CI,
                                                      LoopIDToNumIterationsMap, LoopIDToIncompMap);
          /* insertCodeToComputeConditionalExecutionProbability(CI, CI, IfIDToProbMap); */
          // // NumThreadsInGrid and LoopIDToNumIterationsMap are sufficient to
          // // compute the total number of exectutions of any particular
          // // memory operation in the kernel
          KernelInvocationToInsertionPointMap[CI] = CI;
          auto insertPoint = insertCodeToPerformGlobalMemoryMgmt(CI);
          insertCodeToComputeAccessDensity(insertPoint,
              CI, NumThreadsInGrid, LoopIDToNumIterationsMap, LoopIDToIncompMap,
              KernelInvocationToBDimXMap, KernelInvocationToBDimYMap,
              KernelInvocationToGDimXMap, KernelInvocationToGDimYMap);
        }
        }
      } else {
        errs() << "not iterative kernel\n";
        if (llvm::CallBase *CI = dyn_cast<CallBase>(*KL)) {
          std::map<unsigned, Value *> LoopIDToNumIterationsMap;
        std::map<unsigned, bool> LoopIDToIncompMap;
        std::map<unsigned, Value *> IfIDToProbMap;
          std::map<CallBase *, Value *> KernelInvocationToBDimXMap;
          std::map<CallBase *, Value *> KernelInvocationToBDimYMap;
          std::map<CallBase *, Value *> KernelInvocationToGDimXMap;
          std::map<CallBase *, Value *> KernelInvocationToGDimYMap;
          if(FirstInvocationFound == false) {
              FirstInvocationNonIter = insertPointForFirstInvocationNonIter(CI);
              FirstInvocationFound = true;
          }
          llvm::Value *NumThreadsInGrid = insertCodeToPrintNumThreads(CI,
              CI, KernelInvocationToBDimXMap, KernelInvocationToBDimYMap,
              KernelInvocationToGDimXMap, KernelInvocationToGDimYMap);
          insertCodeToComputeKernelLoopIterationCount(CI, CI,
                                                      LoopIDToNumIterationsMap, LoopIDToIncompMap);
          /* insertCodeToComputeConditionalExecutionProbability(CI, CI, IfIDToProbMap); */
          // // NumThreadsInGrid and LoopIDToNumIterationsMap are sufficient to
          // // compute the total number of exectutions of any particular
          // // memory operation in the kernel
          KernelInvocationToInsertionPointMap[CI] = CI;
          auto insertPoint = insertCodeToPerformGlobalMemoryMgmt(CI);
          insertCodeToComputeAccessDensity(insertPoint,
              CI, NumThreadsInGrid, LoopIDToNumIterationsMap, LoopIDToIncompMap,
              KernelInvocationToBDimXMap, KernelInvocationToBDimYMap,
              KernelInvocationToGDimXMap, KernelInvocationToGDimYMap);
        }
      }
    }

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};

} // namespace

char SCHostTransform::ID = 0;
static RegisterPass<SCHostTransform>
    X("SCHostTransform", "SCHostTransform Pass", true, true);

/*
        traverseExpressionTree(AID->second);
        std::map<ExprTreeNode *, Value *> Unknowns;
        identifyUnknownsFromExpressionTree(CI, Unknowns, AID->second);
        errs() << "\nID unknowns\n";
        for (auto UnknownIter = Unknowns.begin(); UnknownIter != Unknowns.end();
             UnknownIter++) {
          errs() << UnknownIter->first->original_str << "\n";
          (*UnknownIter).second->dump();
        }
        llvm::Value *AccessCountInLoopInKernel =
            insertLoopItersEvaluationCode(CI, Unknowns, AID->second);
        AccessCountInLoopInKernel->dump();
*/
