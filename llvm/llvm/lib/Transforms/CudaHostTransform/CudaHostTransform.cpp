//===- CudaHostTransform.cpp - Extracting access information from CUDA kernels
//---------------===//
//
// We borrow heavily from SC'19 paper
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/Pass.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <stack>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "CudaHostTransform"

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
    ETO_AND,
    ETO_OR,
    ETO_MUL,
    ETO_SHL,
    ETO_PHI,
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
    ETO_INTERM,
    ETO_NONE,
  };

  struct ExprTreeNode {
    ExprTreeOp op;
    /* Value* val; */
    unsigned arg;
    unsigned long long value;
    std::string original_str;
    struct ExprTreeNode* parent;
    struct ExprTreeNode* children[2];
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

  struct SubAllocationStruct{
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
    std::vector<SubAllocationStruct*> SubAllocations;
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

  std::set<Value*> StructAllocas;
  std::map<AllocaInst*, std::map<unsigned, Value*>> StructAllocasToIndexToValuesMap;

  std::set<Function *> ListOfLocallyDefinedFunctions;
  std::map<Function *, std::vector<Value *>> FunctionToFormalArgumentMap;
  std::map<CallBase *, std::vector<Value *>> FunctionCallToActualArumentsMap;
  std::map<Value *, std::vector<Value *>> FormalArgumentToActualArgumentMap;
  /* std::map<Value *, Value *> ActualArgumentToFormalArgumentMap; */
  std::map<Value*, std::map<Value *, Value *>> FunctionCallToFormalArgumentToActualArgumentMap;
  std::map<Value*, std::map<Value *, Value *>> FunctionCallToActualArgumentToFormalArgumentMap;

  std::set<Value *> OriginalPointers;
  std::map<Value *, Value*> PointerOpToOriginalPointers;
  std::map<Value *, Value*> PointerOpToOriginalStructPointer;
  std::map<Value *, unsigned> PointerOpToOriginalStructPointersIndex;

  std::map<Value*, unsigned> PointerOpToOriginalConstant;

  std::set<CallBase*> VisitedCallInstForPointerPropogation;

  std::set<Instruction*> MemcpyOpForStructs;
  std::map<Value*, Instruction*> MemcpyOpForStructsSrcToInstMap;
  std::map<Value*, Instruction*> MemcpyOpForStructsDstToInstMap;

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
  std::map<Value*, std::map<unsigned, unsigned long long>> MallocPointerStructToIndexToSizeMap;

  std::map<Value *, std::vector<Value *>> KernelArgToStoreMap;
  std::map<Instruction *, Value *> KernelInvocationToStructMap;
  std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToActualArgMap;
  std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToAllocationMap;
  std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToLastStoreMap;
  std::map<Instruction*, std::map<Value*, Value*>>
    KernelInvocationToKernArgToAllocationMap;
  std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToConstantMap;
  std::map<Instruction *, std::map<unsigned, Value *>>
    KernelInvocationToArgNumberToLIVMap;
  std::map<Instruction *, std::map<Value*, unsigned>>
    KernelInvocationToLIVToArgNumMap;
  std::map<Instruction *, std::map<BlockSizeType, unsigned>>
    KernelInvocationToBlockSizeMap;
  std::map<Instruction *, std::map<GridSizeType, unsigned>>
    KernelInvocationToGridSizeMap; // when grid size is constant
  std::map<Instruction *, std::map<GridSizeType, Value*>>
    KernelInvocationToGridSizeValueMap; // when grid size is variable


  std::map<Instruction*, unsigned long> KernelInvocationToIterMap;
  std::map<Instruction*, unsigned long> KernelInvocationToStepsMap;

  std::map<Instruction*, std::map<unsigned, unsigned long long>> KernelInvocationToAccessIDToAccessDensity;
  std::map<Instruction*, std::map<unsigned, unsigned>> KernelInvocationToAccessIDToPartDiff_phi;
  std::map<Instruction*, std::map<unsigned, unsigned>> KernelInvocationToAccessIDToPartDiff_bidx;
  std::map<Instruction*, std::map<unsigned, unsigned>> KernelInvocationToAccessIDToPartDiff_bidy;
  std::map<Instruction*, std::map<unsigned, unsigned>> KernelInvocationToAccessIDToPartDiff_looparg;
  std::map<Instruction*, std::map<unsigned, unsigned>> KernelInvocationToAccessIDToWSS;

  std::map<Instruction *, Value*> KernelInvocationToEnclosingLoopMap;
  std::map<Instruction *, Function*> KernelInvocationToEnclosingFunction;

  // map from loop id to loop iterations
  std::map<std::string, std::map<unsigned, std::vector<std::string>>> LoopIDToLoopBoundsMap;
  std::map<std::string, std::map<unsigned, unsigned>> LoopIDToLoopItersMap;

  std::map<std::string, std::map<unsigned, ExprTreeNode*>> LoopIDToBoundsExprMapIn;
  std::map<std::string, std::map<unsigned, ExprTreeNode*>> LoopIDToBoundsExprMapFin;
  std::map<std::string, std::map<unsigned, ExprTreeNode*>> LoopIDToBoundsExprMapStep;
  std::map<std::string, std::map<unsigned, unsigned>> LoopIDToBoundsMapIn;
  std::map<std::string, std::map<unsigned, unsigned>> LoopIDToBoundsMapFin;
  std::map<std::string, std::map<unsigned, unsigned>> LoopIDToBoundsMapStep;


  // map from access id to expression tree
  std::map<std::string, std::map<unsigned, unsigned>> AccessIDToAllocationArgMap;
  std::map<std::string, std::map<unsigned, unsigned>> AccessIDToEnclosingLoopMap;
  std::map<std::string, std::map<unsigned, ExprTreeNode* >> AccessIDToExpressionTreeMap;

  std::set<ExprTreeOp> terminals;
  std::set<ExprTreeOp> operations;

 // terminal values are like kernel-arguments, 
  std::set<Value *> TerminalValues;

  std::map<std::string, std::string> HostSideKernelNameToOriginalNameMap;

  // CudaHostTransform
  struct CudaHostTransform : public ModulePass {
    static char ID; // Pass identification, replacement for typeid

    CudaHostTransform() : ModulePass(ID) {}

    void processMemoryAllocation(CallBase *I) {
      errs() << "processing memory allocation\n";
      I->dump();
      I->getOperand(0) ->dump();
      // I->getOperand(1) ->dump();
      // I->getOperand(1) ->getType()->dump();
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
        // errs() << CI->getSExtValue() << "\n";
        MallocSizeMap[I] = CI->getSExtValue();
        /* MallocPointerToSizeMap[I->getOperand(0)] = CI->getSExtValue(); */
        auto OGPtr = PointerOpToOriginalPointers[I->getOperand(0)];
        if(OGPtr){
          errs()<< "og ptrs = ";
          OGPtr->dump();
          MallocPointerToSizeMap[OGPtr] = CI->getSExtValue();
          if(StructAllocas.find(OGPtr) != StructAllocas.end()){
            errs() << "found struct og ptr\n";
            if(auto GEPI = dyn_cast<GetElementPtrInst>(I->getOperand(0))){
              errs() << "found gepi\n";
              auto numIndices = GEPI->getNumIndices();
              if(numIndices == 2) {
                if(auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(2))){
                  errs() << "og is struct\n";
                  PointerOpToOriginalStructPointersIndex[GEPI] = FieldNum->getSExtValue();
                  errs() << "field num = " << FieldNum->getSExtValue() << "\n";
                  MallocPointerStructToIndexToSizeMap[OGPtr][FieldNum->getSExtValue()] = CI->getSExtValue();
                }
              } else {
                if(auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(1))){
                  errs() << "og maybe struct or array\n";
                  PointerOpToOriginalStructPointersIndex[GEPI] = FieldNum->getSExtValue();
                  errs() << "field num = " << FieldNum->getSExtValue() << "\n";
                  MallocPointerStructToIndexToSizeMap[OGPtr][FieldNum->getSExtValue()] = CI->getSExtValue();
                }
              }
            }
          }
        } else {
          if(FormalArgumentToActualArgumentMap.find(I->getOperand(0)) != FormalArgumentToActualArgumentMap.end()){
            errs() << "found actual arg\n";
            auto actarg = FormalArgumentToActualArgumentMap[I->getOperand(0)][0];
            actarg->dump();
            OGPtr = PointerOpToOriginalPointers[actarg];
            if(OGPtr){
              errs()<< "og ptrs = ";
              OGPtr->dump();
              MallocPointerToSizeMap[OGPtr] = CI->getSExtValue();
              if(StructAllocas.find(OGPtr) != StructAllocas.end()){
                errs() << "found struct og ptr via args\n";
                /* MallocPointerStructToIndexToSizeMap[OGPtr][ */
              }
            }
          }
        }
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
        if(auto *CI = dyn_cast<CallBase>(U)) {
          // if  the passd value is the destination, then return the source
          auto Callee = CI->getCalledFunction();
          if ((Callee && (Callee->getName() == "llvm.memcpy.p0.p0.i64")) ){
            CI->dump();
            if(CI->getOperand(0) == V) {
              errs() << "memcpy call \n";
              return CI->getOperand(1);
            }
          }
        }
      }
      return nullptr;
    }

    StoreInst* findStoreInstWithGivenValueOperand(Value *V) {
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

    // This function identifies the most recent store to the
    void recurseTillStoreOrEmtpy(CallBase *Invocation, Value *KernelArgStruct,
        Value *V, Value* Karg) {
      /* errs() << "recurseTillStoreOrEmpty\n"; */
      V->dump();
      if (auto *SI = dyn_cast<StoreInst>(V)) {
        /* errs() << "STORE\n"; */
        KernelArgToStoreMap[KernelArgStruct].push_back(V);
        /* SI->dump(); */
        int Position = findKernelStructLocationForStoreInstruction(SI); // store location tracing
        Value *Val = findValueForStoreInstruction(SI); // store value tracing
        errs() << "Position in Kernel Arg Struct = " << Position << "\n";
        if (Val) {
          errs() << "Value being written by store operand\n";
          Val->dump();
        }
        if (findStoreInstWithGivenValueOperand(Val)) { // find where the value is coming from
          errs() << "\nFOUND SIWGVO\n";
          findStoreInstWithGivenValueOperand(Val)->dump();
          auto *SIWGPO = findStoreInstWithGivenPointerOperand(Val);
          if(SIWGPO) {
            SIWGPO->dump();
            errs() << "SIWGPO value: ";
            SIWGPO->getValueOperand()->dump();
            if(FormalArgumentToActualArgumentMap.find(SIWGPO) != FormalArgumentToActualArgumentMap.end()){
              errs() << "found SIWGPO as an argument\n";
            }
            if (auto *LIPO = dyn_cast<LoadInst>(SIWGPO->getValueOperand())) {
              errs() << "\nWHICH USES LIPO\n";
              LIPO->getPointerOperand()->dump();
              auto MallocPointer =
                MallocPointerToSizeMap.find(LIPO->getPointerOperand());
              if (MallocPointerToSizeMap.find(MallocPointer->first) !=
                  MallocPointerToSizeMap.end()) {
                /* errs() << "FOUND YAY!!\n"; */
                /* LIPO->getPointerOperand()->dump(); */
                KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                  MallocPointer->first;
                KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                  MallocPointer->first;
                KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                  MallocPointer->first;
                KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] = SI;
                // LIPO->getPointerOperand();
              }
              auto ArgumentPointer = LIPO->getPointerOperand();
              if (FormalArgumentToActualArgumentMap.find(ArgumentPointer) != FormalArgumentToActualArgumentMap.end()){
                errs() << "Found LIPO as a formal argument\n";
                auto Ptr = FormalArgumentToActualArgumentMap[ArgumentPointer][0];
                Ptr->dump();
                if(PointerOpToOriginalPointers.find(Ptr) != PointerOpToOriginalPointers.end()){
                  errs() << "found OG pointer\n";
                  PointerOpToOriginalPointers[Ptr]->dump();
              auto MallocPointer = PointerOpToOriginalPointers.find(Ptr);
                KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                  MallocPointer->second;
                KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                  MallocPointer->second;
                KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                  MallocPointer->second;
                KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] = SI;
                }
              }
            }
            if (isa<PointerType>(SIWGPO->getValueOperand()->getType())) {
              errs() << "\n WHICH IS A POINTER\n";
              auto MallocPointer =
                PointerOpToOriginalPointers.find(SIWGPO->getValueOperand());
              if (MallocPointer != PointerOpToOriginalPointers.end()) {
                /* MallocPointer->first->dump(); */
                /* MallocPointer->second->dump(); */
                /* errs() << "FOUND YAY!!\n"; */
                /* LIPO->getPointerOperand()->dump(); */
                KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                  MallocPointer->first;
                KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                  MallocPointer->first;
                KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                  MallocPointer->first;
                KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] = SI;
              }
            }
            if (auto *CIPO = dyn_cast<ConstantInt>(SIWGPO->getValueOperand())) {
              /* errs() << "constant\n"; */
              /* CIPO->dump(); */
              KernelInvocationToArgNumberToConstantMap[Invocation][Position] = CIPO;
              KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                CIPO;
            }
            if (KernelInvocationToEnclosingLoopMap[Invocation] == SIWGPO->getValueOperand()) {
              errs() << "host loop\n";
              Value* LIV = SIWGPO->getValueOperand();
              LIV->dump();
              KernelInvocationToArgNumberToLIVMap[Invocation][Position] = LIV;
              KernelInvocationToLIVToArgNumMap[Invocation][LIV] = Position;
              KernelInvocationToArgNumberToActualArgMap[Invocation][Position] = LIV;
            }
            KernelInvocationToArgNumberToActualArgMap[Invocation][Position] = SIWGPO->getValueOperand();
          } else {
            errs() << "complicated case\n";
            Val->dump();
            // iterate over all mempcy for structs and find out if any match Val
            if(MemcpyOpForStructsDstToInstMap.find(Val) != MemcpyOpForStructsDstToInstMap.end()) {
              errs() << "found writer\n";
              auto memcpyInst = MemcpyOpForStructsDstToInstMap[Val];
              memcpyInst->dump();
              errs() << "source\n";
              auto src = memcpyInst->getOperand(1);
              src->dump();
              auto MallocPointer = PointerOpToOriginalPointers.find(src);
              if (MallocPointer != PointerOpToOriginalPointers.end()) {
                errs() << "original pointer\n";
                MallocPointer->second->dump();
                KernelInvocationToArgNumberToAllocationMap[Invocation][Position] =
                  MallocPointer->second;
                KernelInvocationToArgNumberToActualArgMap[Invocation][Position] =
                  MallocPointer->second;
                KernelInvocationToKernArgToAllocationMap[Invocation][Karg] =
                  MallocPointer->second;
                KernelInvocationToArgNumberToLastStoreMap[Invocation][Position] = SI;
              }
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
      /* errs() << "host side name : " << HostSideKernelNameToOriginalNameMap[Mangledname] << "\n"; */
      return HostSideKernelNameToOriginalNameMap[Mangledname];
      /* return Mangledname.erase(0, 19); */
    }

    bool isNumber(std::string op) {
      bool isNum = true;
      for(int i = 0; i < op.length(); i++) {
        if(op[0] == '-'){
          continue;
        }
        if(!isDigit(op[i])) {
          isNum = false;
          break;
        }
      }
      return isNum;
    }

    ExprTreeOp getExprTreeOp(std::string op) {
      if(isNumber(op)) {
        return ETO_CONST;
      }
      if(op.compare("PC") == 0) {
        return ETO_PC;
      } else if(op.compare("ADD") == 0) {
        return ETO_ADD;
      } else if(op.compare("AND") == 0) {
        return ETO_AND;
      } else if(op.compare("OR") == 0) {
        return ETO_OR;
      } else if(op.compare("MUL") == 0) {
        return ETO_MUL;
      } else if(op.compare("SHL") == 0) {
        return ETO_SHL;
      } else if(op.compare("PHI") == 0) {
        return ETO_PHI;
      } else if(op.compare("LOAD") == 0) {
        return ETO_MEMOP;
      } else if(op.compare("STORE") == 0) {
        return ETO_MEMOP;
      } else if(op.compare("TIDX") == 0) {
        return ETO_TIDX;
      } else if(op.compare("TIDY") == 0) {
        return ETO_TIDY;
      } else if(op.compare("BIDX") == 0) {
        return ETO_BIDX;
      } else if(op.compare("BIDY") == 0) {
        return ETO_BIDY;
      } else if(op.compare("BDIMX") == 0) {
        return ETO_BDIMX;
      } else if(op.compare("BDIMY") == 0) {
        return ETO_BDIMY;
      } else if(op.substr(0, 3).compare("ARG") == 0) {
        return ETO_ARG;
      }
      return ETO_NONE;
    }

    unsigned getExprTreeNodeArg(std::string op) {
      return stoi(op.substr(3));
    }

    bool isTerminal(ExprTreeNode* node) {
      /* errs() << node->op << "\n"; */
      if(terminals.find(node->op) != terminals.end()){
        return true;
      }
      return false;
    }

    bool isPhiNode(ExprTreeNode* node) {
      if(node->op == ETO_PHI)
        return true;
      return false;
    }

    bool isOperation(ExprTreeNode* node) {
      if(operations.find(node->op) != operations.end()){
        return true;
      }
      return false;
    }

    void traverseExpressionTree(ExprTreeNode* root) {
      if(root == nullptr) return;
      /* errs() << "\ntraverse expression tree\n"; */
      std::stack<ExprTreeNode*> Stack;
      Stack.push(root);
      while (!Stack.empty()) {
        ExprTreeNode *Current = Stack.top();
        errs() << Current->original_str << " ";
        Stack.pop();
        if(isOperation(Current)){
          Stack.push(Current->children[0]);
          Stack.push(Current->children[1]);
        }
      }
    }

    ExprTreeNode* findNodeInExpressionTree(ExprTreeNode* root, ExprTreeOp op, unsigned arg) {
      /* errs() << "\nfind node in expression tree\n"; */
      std::stack<ExprTreeNode*> Stack;
      Stack.push(root);
      while (!Stack.empty()) {
        ExprTreeNode *Current = Stack.top();
        if(Current->op == op) {
          if(op == ETO_ARG) {
            if( Current->arg == arg){
              return Current;
            }
          } else {
            return Current;
          }
        }
        /* errs() << Current->original_str << " "; */
        Stack.pop();
        if(isOperation(Current)){
          Stack.push(Current->children[0]);
          Stack.push(Current->children[1]);
        }
      }
      return nullptr;
    }

    unsigned getMaxValueForLiterals(CallBase *CI, ExprTreeNode* node, unsigned LoopArg, unsigned loopid) {
      if(node->op == ETO_ARG) {
        if(node->arg == LoopArg) {
          // TODO:  get loop bounds
          return 0;
        } else {
          return getActualHostValueForLiterals(CI, node);
        }
      }
      if(node->op == ETO_PHI_TERM) {
        // get lower bound of phi terminal from loop information file
        auto *KernelPointer = CI->getArgOperand(0);
        auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
        auto KernelName = KernelFunction->getName();
        std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
        if(loopid == 0) {
          return 1;
        }
        auto Fin = LoopIDToBoundsExprMapFin[OriginalKernelName][loopid];
        return evaluateExpressionTree(CI, Fin);
      }
      if(node->op == ETO_BIDX || node->op == ETO_BIDY ||
          node->op == ETO_TIDX || node->op == ETO_TIDY) {
        if(node->op == ETO_TIDX) {
          return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX] - 1;
        }
        if(node->op == ETO_TIDY) {
          return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY] - 1;
        }
        if(node->op == ETO_BIDX) {
          if(KernelInvocationToGridSizeValueMap[CI].find(AXIS_TYPE_GDIMX) != KernelInvocationToGridSizeValueMap[CI].end()) {
            auto GridValueSize = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
            auto RRPN = getExpressionTree(GridValueSize);
            auto griddimx = evaluateRPNForIter0(CI, RRPN);
            return griddimx; 
          } else {
            errs() << "hehe: " << KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX] << "\n";
            return KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX] - 1;
          }
        }
        if(node->op == ETO_BIDY) {
          if(KernelInvocationToGridSizeValueMap[CI].find(AXIS_TYPE_GDIMY) != KernelInvocationToGridSizeValueMap[CI].end()) {
            auto GridValueSize = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY];
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

    unsigned getMinValueForLiterals(CallBase *CI, ExprTreeNode* node, unsigned LoopArg, unsigned loopid) {
      if(node->op == ETO_ARG) {
        if(node->arg == LoopArg) {
          // TODO:  get loop bounds
          return 0;
        } else {
          return getActualHostValueForLiterals(CI, node);
        }
      }
      if(node->op == ETO_PHI_TERM) {
        // get lower bound of phi terminal from loop information file
        auto *KernelPointer = CI->getArgOperand(0);
        auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
        auto KernelName = KernelFunction->getName();
        std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
        if(loopid == 0) {
          return 1;
        }
        auto In = LoopIDToBoundsExprMapIn[OriginalKernelName][loopid];
        return evaluateExpressionTree(CI, In);
      }
      if(node->op == ETO_BIDX || node->op == ETO_BIDY ||
          node->op == ETO_TIDX || node->op == ETO_TIDY) {
        return 0;
      }
      return getActualHostValueForLiterals(CI, node);
    }

    unsigned getActualHostValueForLiterals(CallBase *CI, ExprTreeNode* node) {
      if(node->op == ETO_INTERM) {
        return node->value;
      }
      if(node->op == ETO_CONST) {
        return stoi(node->original_str);
      }
      if(node->op == ETO_BDIMX) {
        return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
      }
      if(node->op == ETO_BDIMY) {
        return KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
      }
      if(node->op == ETO_BIDX || node->op == ETO_BIDY) {
        return 1;
      }
      if(node->op == ETO_ARG) {
        std::map<unsigned, Value*> ArgNumToConstantMap = KernelInvocationToArgNumberToConstantMap[CI];
        if(ArgNumToConstantMap.find(node->arg) != ArgNumToConstantMap.end()){
          Value* ConstArg = ArgNumToConstantMap[node->arg];
          if (ConstantInt *CI = dyn_cast<ConstantInt>(ConstArg)) {
            return CI->getSExtValue();
          }
          return 0;
        }
        // here is to add constant arguments from host side used as arguments to kernel side
        Value* ArgInQ = KernelInvocationToArgNumberToActualArgMap[CI][node->arg];
        if(ArgInQ) {
          /* CI->dump(); */
          /* errs() << "ARG IN Q\n"; */
          /* ArgInQ->dump(); */
          /* errs() << "actual arg\n"; */
          /* FunctionCallToFormalArgumentToActualArgumentMap[CI][ArgInQ]->dump(); */
          // we are assuming there is only one call to each function containing a kernel invocation.
          /* FormalArgumentToActualArgumentMap[ArgInQ][0]->dump(); */
          if(auto ConstI = dyn_cast<ConstantInt>(FormalArgumentToActualArgumentMap[ArgInQ][0])){
            return ConstI->getSExtValue();
          }
          // MAJOR TODO: ArgInQ may have to be traversed in reverse till it becomes:function arg
        }
      }
      // so on and so forth
      // get constant args
      return 0;
    }

    ExprTreeNode* operateMax(CallBase* CI, ExprTreeNode* operation, ExprTreeNode* op1, ExprTreeNode* op2, unsigned LoopArg, unsigned loopid) {
      ExprTreeNode* result = new ExprTreeNode();
      unsigned long long v1 = getMaxValueForLiterals(CI, op1, LoopArg, loopid);
      unsigned long long v2 = getMaxValueForLiterals(CI, op2, LoopArg, loopid);
      unsigned long long res = 1;
      errs() << operation->original_str << "::::" << v1 << " " << v2 << "\n";
      if(operation->op == ETO_SHL) {
        res = v1 << v2;
      }
      if(operation->op == ETO_MUL) {
        res = v1 * v2;
      }
      if(operation->op == ETO_ADD) {
        res = v1 + v2;
      }
      if(operation->op == ETO_OR) {
        res = v1 + v2;
      }
      if(operation->op == ETO_PHI) {
        res = (v1 < v2) ? v2 : v1;
      }
      result->op = ETO_INTERM;
      result->value = res;
      return result;
    }

    ExprTreeNode* operateMin(CallBase* CI, ExprTreeNode* operation, ExprTreeNode* op1, ExprTreeNode* op2, unsigned LoopArg, unsigned loopid) {
      ExprTreeNode* result = new ExprTreeNode();
      unsigned long long v1 = getMinValueForLiterals(CI, op1, LoopArg, loopid);
      unsigned long long v2 = getMinValueForLiterals(CI, op2, LoopArg, loopid);
      unsigned long long res = 1;
      /* errs() << operation->original_str << "::::" << v1 << " " << v2 << "\n"; */
      if(operation->op == ETO_SHL) {
        res = v1 << v2;
      }
      if(operation->op == ETO_MUL) {
        res = v1 * v2;
      }
      if(operation->op == ETO_ADD) {
        res = v1 + v2;
      }
      if(operation->op == ETO_OR) {
        res = v1 + v2;
      }
      if(operation->op == ETO_PHI) {
        res = (v1 < v2) ? v1 : v2;
      }
      result->op = ETO_INTERM;
      result->value = res;
      return result;
    }

    ExprTreeNode* operate(CallBase* CI, ExprTreeNode* operation, ExprTreeNode* op1, ExprTreeNode* op2) {
      ExprTreeNode* result = new ExprTreeNode();
      unsigned long long v1 = getActualHostValueForLiterals(CI, op1);
      unsigned long long v2 = getActualHostValueForLiterals(CI, op2);
      unsigned long long res = 1;
      /* errs() << v1 << " " << v2 << "\n"; */
      if(operation->op == ETO_SHL) {
        res = v1 << v2;
      }
      if(operation->op == ETO_MUL) {
        res = v1 * v2;
      }
      if(operation->op == ETO_ADD) {
        res = v1 + v2;
      }
      if(operation->op == ETO_OR) {
        res = v1 + v2;
      }
      result->op = ETO_INTERM;
      result->value = res;
      return result;
    }

    unsigned long long evaluateRPNforMax(CallBase* CI, std::vector<ExprTreeNode*> RPN, unsigned LoopArg, unsigned loopid) {
      errs() << "Evaluating RPN for max\n";
      std::stack<ExprTreeNode*> stack;
      for (auto Token = RPN.begin(); Token != RPN.end(); Token++){
        errs() << (*Token)->original_str << "\n ";
        if(isOperation(*Token)){
          /* errs() << "operation\n"; */
          ExprTreeNode* op1 = stack.top();
          stack.pop();
          ExprTreeNode* op2 = stack.top();
          stack.pop();
          // evaluate the expression 
          ExprTreeNode* result ;
          if(isTerminal(op1) && isTerminal(op2)){
            result = operateMax(CI, *Token, op1, op2, LoopArg, loopid);
            errs() << "interm = " << result->value << "\n";
          } else{
            errs() << "MAJOR ISSUE: node not teminal\n";
          }
          result->op = ETO_INTERM;
          stack.push(result);
          /* errs() << "TOS = " << result->value << "\n"; */
        }
        else {
          /* errs() << "terminal\n"; */
          (*Token)->value = getMaxValueForLiterals(CI, (*Token), LoopArg, loopid);
          stack.push((*Token));
        }
      }
      return stack.top()->value;
    }


    unsigned long long evaluateRPNforMin(CallBase* CI, std::vector<ExprTreeNode*> RPN, unsigned LoopArg, unsigned loopid) {
      /* errs() << "Evaluating RPN for min\n"; */
      std::stack<ExprTreeNode*> stack;
      for (auto Token = RPN.begin(); Token != RPN.end(); Token++){
        /* errs() << (*Token)->original_str << ": "; */
        if(isOperation(*Token)){
          /* errs() << "operation\n"; */
          ExprTreeNode* op1 = stack.top();
          stack.pop();
          ExprTreeNode* op2 = stack.top();
          stack.pop();
          // evaluate the expression 
          ExprTreeNode* result ;
          if(isTerminal(op1) && isTerminal(op2)){
            result = operateMin(CI, *Token, op1, op2, LoopArg, loopid);
            /* errs() << "interm = " << result->value << "\n"; */
          } else{
            errs() << "MAJOR ISSUE: node not teminal\n";
          }
          result->op = ETO_INTERM;
          stack.push(result);
          /* errs() << "TOS = " << result->value << "\n"; */
        }
        else {
          /* errs() << "terminal\n"; */
          (*Token)->value = getMinValueForLiterals(CI, (*Token), LoopArg, loopid);
          stack.push((*Token));
        }
      }
      return stack.top()->value;
    }

    unsigned long long evaluateRPN(CallBase* CI, std::vector<ExprTreeNode*> RPN) {
      /* errs() << "Evaluating RPN\n"; */
      std::stack<ExprTreeNode*> stack;
      for (auto Token = RPN.begin(); Token != RPN.end(); Token++){
        /* errs() << (*Token)->original_str << ": "; */
        if(isOperation(*Token)){
          /* errs() << "operation\n"; */
          ExprTreeNode* op1 = stack.top();
          stack.pop();
          ExprTreeNode* op2 = stack.top();
          stack.pop();
          // evaluate the expression 
          ExprTreeNode* result ;
          if(isTerminal(op1) && isTerminal(op2)){
            result = operate(CI, *Token, op1, op2);
            /* errs() << "interm = " << result->value << "\n"; */
          } else{
            errs() << "MAJOR ISSUE: node not teminal\n";
          }
          result->op = ETO_INTERM;
          stack.push(result);
        }
        else {
          /* errs() << "terminal\n"; */
          (*Token)->value = getActualHostValueForLiterals(CI, (*Token));
          stack.push((*Token));
        }
      }
      return stack.top()->value;
    }

    // substitute values from host side to get concrete values
    unsigned long long evaluateExpressionTree(CallBase* CI, ExprTreeNode* root) {
      /* errs() << "evaluating expr rooted at " << root->original_str << "\n"; */
      // Convert to RPN
      std::stack<ExprTreeNode*> Stack;
      std::vector<ExprTreeNode*> RPN;
      Stack.push(root);
      while (!Stack.empty()) {
        ExprTreeNode *Current = Stack.top();
        /* errs() << Current->original_str << " "; */
        Stack.pop();
        RPN.push_back(Current);
        if(isOperation(Current)){
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

    std::vector<ExprTreeNode*> findMultipliersByTraversingUpExptTree(ExprTreeNode* root, ExprTreeNode *given) {
      errs() << "\nfind multipliers\n";
      std::vector<ExprTreeNode*> Multipliers;
      ExprTreeNode* current = given;
      ExprTreeNode* parent = current->parent;
      while(current != nullptr) {
        parent = current->parent;
        if(parent){
          if(parent->op == ETO_MUL || parent->op == ETO_SHL) {
            errs() << parent->original_str <<  "  "  << parent->op << "\n";
            if(parent->children[0] == current) {
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
    // Further, we only handle cases where the values across iteratins are of the form C*i, 
    // where i is the induction and C is a  
    unsigned long long partialDifferenceWRTPhi(CallBase* CI, ExprTreeNode* root) {
      errs() << "\npartial diff with rt phi\n";
      ExprTreeNode* phi = findNodeInExpressionTree(root, ETO_PHI, 0);
      ExprTreeNode* phiterm = findNodeInExpressionTree(root, ETO_PHI_TERM, 0);
      std::vector<ExprTreeNode*> Adders;
      ExprTreeNode* current = phiterm;
      ExprTreeNode* parent;
      if(phi  && phiterm){
        errs() << phi->original_str << "\n";
        errs() << phiterm->original_str << "\n";
        while(current != phi){
          parent = current->parent;
          if(parent->children[0] == current) {
            Adders.push_back(parent->children[1]);
          }
          if(parent->children[1] == current) {
            Adders.push_back(parent->children[0]);
          }
          current = parent;
        }
        errs() << "partial difference wrt phi node\n";
        unsigned long long partialDiffWRTPhi = evaluateExpressionTree(CI, Adders[0]);
        return partialDiffWRTPhi;
      }
      return 0;
    }

    unsigned partialDifferenceOfExpressionTreeWRTGivenNode(CallBase*CI, ExprTreeNode* root, ExprTreeOp given, unsigned arg){
      // Find the given op in the expression tree
      // TODO: fin all nodes of the given type in the expression tree
      ExprTreeNode* node = findNodeInExpressionTree(root, given, arg);
      if(node) errs() << "found node " << node->original_str << "\n";
      else errs() << "not found node \n";
      // move up towards root, looking for MUL/SHLs. Store the other child.
      if(node) {
        auto mutlipliers = findMultipliersByTraversingUpExptTree(root, node);
        errs() << "multipliers => ";
        for(auto mutliplier = mutlipliers.begin(); mutliplier != mutlipliers.end(); mutliplier++) {
          errs() << (*mutliplier)->original_str << ".";
        }

        // evaluate the stored other childs and multiply all of them together
        unsigned long long FinalMultiplier = 1;
        for(auto mutliplier = mutlipliers.begin(); mutliplier != mutlipliers.end(); mutliplier++) {
          if((*mutliplier)->parent->op == ETO_MUL) {
            FinalMultiplier *= evaluateExpressionTree(CI, (*mutliplier));
            errs() << "finmul = " << FinalMultiplier << "\n";
          }
          if((*mutliplier)->parent->op == ETO_SHL) {
            FinalMultiplier = FinalMultiplier << evaluateExpressionTree(CI, (*mutliplier));
            errs() << "finmul = " << FinalMultiplier << "\n";
          }
        }
        return FinalMultiplier;
      }

      return 0;
    }

    ExprTreeNode* createExpressionTree(std::vector<std::string> RPN) {
      ExprTreeNode *root = nullptr;
      ExprTreeNode *current = nullptr;
      std::stack<ExprTreeNode*> stack;
      std::vector<ExprTreeNode*> RPN_Nodes;
      /* errs() << "\ncreate expression tree\n"; */
      reverse(RPN.begin(), RPN.end());
      if(RPN.size() == 0) {
        return nullptr;
      }
      for (auto str = RPN.begin(); str != RPN.end(); str++){
        /* errs() << *str << "\n"; */
        current = new ExprTreeNode();
        current->op = getExprTreeOp(*str);
        current->original_str = *str;
        current->parent = nullptr;
        RPN_Nodes.push_back(current);
      }
      for (auto node = RPN_Nodes.begin(); node != RPN_Nodes.end(); node++){
        if((*node)->op == ETO_ARG) {
          /* errs() << (*node)->original_str << " " ; */
          (*node)->arg = getExprTreeNodeArg((*node)->original_str);
          /* errs() << (*node)->arg << " "; */
        }
      }
      // make tree using stack
      /* errs() << "\nmaking tree\n"; */
      bool phi_term_seen = false;
      for (auto node = RPN_Nodes.begin(); node != RPN_Nodes.end(); node++){
        errs() << "\n" << (*node)->original_str << " " ;
        if(isPhiNode(*node) ){ // first phi node is term, second and later are ops: FIXME
          if(phi_term_seen == false ) {
            /* errs() << "Terminal PHI"; */
            (*node)->op = ETO_PHI_TERM;
            stack.push(*node);
            phi_term_seen = true;
          } else {
            /* errs() << "Operation PHI"; */
            if(stack.empty()) return nullptr;
            ExprTreeNode* child1 = stack.top();
            stack.pop();
            if(stack.empty()) return nullptr;
            ExprTreeNode* child2 = stack.top();
            stack.pop();
            child1->parent = *node;
            child2->parent = *node;
            (*node)->children[0] = child1;
            (*node)->children[1] = child2;
            stack.push(*node);
          }
        } else if(isOperation(*node)) {
          errs() << "Operation ";
            if(stack.empty()) return nullptr;
          ExprTreeNode* child1 = stack.top();
          stack.pop();
            if(stack.empty()) return nullptr;
          ExprTreeNode* child2 = stack.top();
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

    void printKernelDeviceAnalyis() {
      std::string Data;

      std::ifstream LoopDetailFile("loop_detail_file.lst");
      if(LoopDetailFile.is_open()){
        errs() << "Reading Loop Detail File\n";
        std::string line;
        while (getline(LoopDetailFile, line)){
          std::stringstream ss(line);
          std::string word;
          ss >> word;
          /* word.erase(0,4); */
          std::string KernelName = word;
          errs() << KernelName << " ";
          ss >> word;
          unsigned LoopId = stoi(word);
          /* errs() << LoopId << " "; */
          std::vector<std::string> IN, FIN, STEP;
          int in = 0, fin = 0, step = 0, iters = 0;
          while(ss >> word) {
            /* errs() << word << " "; */
            if(word.compare("IT") == 0){
              ss >> word;
              iters = stoi (word);
              /* errs() << iters; */
              LoopIDToLoopItersMap[KernelName][LoopId] = iters;
            }
            else {
              /* errs() << word << " "; */
              LoopIDToLoopBoundsMap[KernelName][LoopId].push_back(word);
            }
          }
          errs() << "\n";
        }
      }

      std::ifstream AccessDetailFile("access_detail_file.lst");
      if(AccessDetailFile.is_open()){
        errs() << "Reading Access Detail File\n";
        std::string line;
        while (getline(AccessDetailFile, line)){
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
          AccessIDToAllocationArgMap[KernelName][AccessId] = ParamNumber;
          ss >> word;
          unsigned LoopId = stoi(word);
          errs() << LoopId << " ";
          AccessIDToEnclosingLoopMap[KernelName][AccessId] = LoopId;
          std::vector<std::string> RPN;
          while(ss >> word) {
            errs() << word << " ";
            if(word.compare("[") == 0 || word.compare("]") == 0){
              ;
            }
            else {
              RPN.push_back(word);
            }
          }
          errs() << "good expression\n";
          AccessIDToExpressionTreeMap[KernelName][AccessId] = createExpressionTree(RPN);
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

    void traverseGridSizeArgument(Value* GridSizeArgument) {
      errs() << "traverse grid size arg\n";
      GridSizeArgument->dump();
    }

    void parseGridSizeArgument(Value* GridSizeArgument, CallBase* CI) {
      errs() << "parsing grid size argrument\n";
      GridSizeArgument->dump();
      if(auto GridSizeOp = dyn_cast<Instruction> (GridSizeArgument)){
        if(GridSizeOp->getOpcode() == Instruction::Mul){
          errs() << "MUL\n";
          for(auto &Operand : GridSizeOp->operands()){
            /* Operand->dump(); */
            if(auto ConstOper = dyn_cast<ConstantInt>(Operand)){
              /* errs() << "is constant\n"; */
              /* errs() << ConstOper->getSExtValue() << "\n"; */
              if(ConstOper->getSExtValue() == 4294967297) {
                errs() << "magic duplication operation\n";
                for(auto &OtherOpCandidate : GridSizeOp->operands()){
                  if(OtherOpCandidate != ConstOper) {
                    traverseGridSizeArgument(OtherOpCandidate);
                    KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX] = OtherOpCandidate;
                    KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY] = OtherOpCandidate;
                  }
                }
              }
            }
          }
        }
        if(GridSizeOp->getOpcode() == Instruction::Or){
          errs() << "OR\n";
          for(auto &Operand : GridSizeOp->operands()){
            /* Operand->dump(); */
            if(auto ConstOper = dyn_cast<ConstantInt>(Operand)){
              /* errs() << "is constant\n"; */
              /* errs() << ConstOper->getSExtValue() << "\n"; */
              if(ConstOper->getSExtValue() == 4294967296) { // hard coded. ideally use arithemtic to figure
                errs() << "magic operation\n";
                for(auto &OtherOpCandidate : GridSizeOp->operands()){
                  if(OtherOpCandidate != ConstOper) {
                    errs() << "magic operation pushed\n";
                    CI->dump();
                    traverseGridSizeArgument(OtherOpCandidate);
                    KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX] = OtherOpCandidate;
                    KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY] = 1; // hardcoded.
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
        if (auto *GridXYConst = dyn_cast<ConstantInt>(GridXYValue)) {
          unsigned long long GridXY = GridXYConst->getSExtValue();
          GridDimY = GridXY >> 32;
          GridDimX = (GridXY << 32) >> 32;
          errs() << "Grid X = " << GridDimX << "\n";
          errs() << "Grid Y = " << GridDimY << "\n";
          KernelInvocationToGridSizeMap[LaunchCall[Index]][AXIS_TYPE_GDIMX] = GridDimX;
          KernelInvocationToGridSizeMap[LaunchCall[Index]][AXIS_TYPE_GDIMY] = GridDimY;
        } else {
          errs() << "heh\n";
          parseGridSizeArgument(GridXYValue, LaunchCall[Index]);
        }
        Value *GridZValue = PushCall[Index]->getOperand(1);
        if (auto *GridZConst = dyn_cast<ConstantInt>(GridZValue)) {
          unsigned long GridZ = GridZConst->getSExtValue();
          GridDimZ = GridZ;
          errs() << "Grid Z = " << GridDimZ << "\n";
          KernelInvocationToGridSizeMap[LaunchCall[Index]][AXIS_TYPE_GDIMZ] = GridDimZ;
        } else {
          static_assert(true, "NO reach here. GRID DIM must be constant \n");
        }
        GridZValue->dump();
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
      for (llvm::User *Karg : KernelArgs->users()) {
        errs() << "user: ";
        Karg->dump();
        recurseTillStoreOrEmtpy(I, KernelArgs, Karg, Karg);
      }
      // for (auto &Arg: I->args()){
      //   Arg->dump();
      // }
      return;
    }

    unsigned computeLoopIterations(CallBase *CI, std::string kernelName, int loopID){
      /* errs() << "\nCompute Loop Iterations" << loopID << "\n"; */
      if(loopID == 0) 
        return 1;
      std::map<unsigned, unsigned> kernelLoopToIterMap = LoopIDToLoopItersMap[kernelName];
      if(kernelLoopToIterMap.find(loopID) != kernelLoopToIterMap.end()){
        return kernelLoopToIterMap[loopID];
      }
      std::map<unsigned, std::vector<std::string>> kernelLoopToBoundsMap = LoopIDToLoopBoundsMap[kernelName];
      if(kernelLoopToBoundsMap.find(loopID) != kernelLoopToBoundsMap.end()){
        // compute the loop bound after substituting actual values
        std::vector<std::string> LoopBoundsTokens = kernelLoopToBoundsMap[loopID];
        /* errs() << "\nloop tokens\n"; */
        ExprTreeNode *In, *Fin, *Step;
        unsigned long long InVal, FinVal, StepVal;
        std::vector<std::string> SplitTokens[3];
        unsigned currentSplit = 0;
        for(auto Token = LoopBoundsTokens.begin(); Token != LoopBoundsTokens.end(); Token++) {
          /* errs() << *Token << " "; */
          if((*Token).compare("IN") == 0) {
            currentSplit = 0;
          } else if((*Token).compare("FIN") == 0) {
            currentSplit = 1;
          } else if((*Token).compare("STEP") == 0) {
            currentSplit = 2;
          } else {
            SplitTokens[currentSplit].push_back(*Token);
          }
        }
        /* errs() << "\nIn\n"; */
        In = createExpressionTree(SplitTokens[0]);
        if(In == nullptr) {
          InVal = 0;
        } else {
          InVal = evaluateExpressionTree(CI, In);
        }
        LoopIDToBoundsExprMapIn[kernelName][loopID] = In;
        /* errs() << InVal << "\n"; */
        /* errs() << "\nFin\n"; */
        Fin = createExpressionTree(SplitTokens[1]);
        if(Fin == nullptr) {
          FinVal = 1;
        } else {
          FinVal = evaluateExpressionTree(CI, Fin);
        }
        LoopIDToBoundsExprMapFin[kernelName][loopID] = Fin;
        /* errs() << FinVal << "\n"; */
        /* errs() << "\nStep\n"; */
        Step = createExpressionTree(SplitTokens[2]);
        if(Step == nullptr) {
          StepVal = 1;
        } else {
          StepVal = evaluateExpressionTree(CI, Step);
        }
        LoopIDToBoundsExprMapStep[kernelName][loopID] = Step;
        /* errs() << StepVal << "\n"; */
        /* errs() << "\n"; */
        unsigned iters = 1;
        iters = (FinVal - InVal)/StepVal;
        return iters;
      }
      // else compute the loop iterations from data from device analysis
      return 1;
    }

    unsigned long int getAllocationSize(Value* PointerOp) {
      errs() << "get alloation size (pointer) \n";
      PointerOp->dump();
      auto *OriginalPointer = PointerOpToOriginalPointers[PointerOp];
      OriginalPointer->dump();
      if(StructAllocas.find(OriginalPointer) != StructAllocas.end()){
        if(PointerOpToOriginalStructPointersIndex.find(PointerOp) != PointerOpToOriginalStructPointersIndex.end()){
          auto argnum =  PointerOpToOriginalStructPointersIndex[PointerOp];
          errs() << "faund: " << argnum << "\n";
          unsigned long int AllocationSize = MallocPointerStructToIndexToSizeMap[OriginalPointer][argnum];
          return AllocationSize;
        }
      } 
      unsigned long int AllocationSize = MallocPointerToSizeMap[OriginalPointer];
      return AllocationSize;
    }

    unsigned long int getAllocationSize(CallBase *CI, unsigned argid) {
      errs() << "get alloation size\n";
      auto ArgNumberToAllocationMap = KernelInvocationToArgNumberToAllocationMap[CI];
      auto PointerOp = ArgNumberToAllocationMap[argid];
      PointerOp->dump();
      auto *OriginalPointer = PointerOpToOriginalPointers[PointerOp];
      OriginalPointer->dump();
      if(StructAllocas.find(OriginalPointer) != StructAllocas.end()){
      errs() << "faund: " << PointerOpToOriginalStructPointersIndex[PointerOp] << "\n";
      }
      unsigned long int AllocationSize = MallocPointerToSizeMap[OriginalPointer];
      return AllocationSize;
    }

    long long operate(BinaryOperator* BO, int v1, int v2) {
      long long res;
      /* errs() << v1 << "  " << v2 << "\n"; */
      if(BO->getOpcode() == Instruction::Mul) {
        res = v1 * v2;
      }
      if(BO->getOpcode() == Instruction::SDiv) {
        res = v2 / v1;
      }
      if(BO->getOpcode() == Instruction::Sub) {
        res = v2 - v1;
      }
      if(BO->getOpcode() == Instruction::Add) {
        res = v1 + v2;
      }
      if(BO->getOpcode() == Instruction::LShr) {
        res = v2 >> v1;
      }
      /* errs() << "resu = " << res << "\n"; */
      return res;
    }

    // iter 0 has phi node with value initial
     long long evaluateRPNForIter0(CallBase* CI, std::vector<Value*> RPN) {
      /* std::vector<Value*> RPN = RRPN; */
      reverse(RPN.begin(), RPN.end());
      bool phiseen = false;

      std::stack<long long> stack;

      for (auto Token = RPN.begin(); Token != RPN.end(); Token++){
        // if terminal (host side terminals only include function arguments)
        (*Token)->dump();
        if(TerminalValues.find(*Token) != TerminalValues.end()) {
          auto ActualArg = FormalArgumentToActualArgumentMap[*Token][0];
          ActualArg->dump();
          if(ConstantInt *CoI = dyn_cast<ConstantInt>(ActualArg)) {
            /* errs() << "Yayaya " << CoI->getSExtValue() << "\n"; */
            stack.push(CoI->getSExtValue());
            /* errs() << "stack push = " << CoI->getSExtValue() << "\n"; */
          } else {
            /* errs() << "PANIC!!!!\n"; */
            errs() << "NOt a constant, so checking for values\n";
            if(PointerOpToOriginalConstant.find(ActualArg) != PointerOpToOriginalConstant.end()){
              errs() << PointerOpToOriginalConstant[ActualArg] << "\n";
              stack.push(PointerOpToOriginalConstant[ActualArg]);
            }
          }
          continue;
        }
        if(ConstantInt* CoI = dyn_cast<ConstantInt>(*Token)){
          stack.push(CoI->getSExtValue());
          /* errs() << "stack push = " << CoI->getSExtValue() << "\n"; */
          continue;
        }
        if(Instruction* I = dyn_cast<Instruction>(*Token)){
          // order matters, probably
          if(PHINode *Phi = dyn_cast<PHINode>(I)){
            if( phiseen == false){
              stack.push(0); // 
              /* errs() << "stack push = 0\n"; */
              phiseen = true;
              continue;
            }
            if( phiseen == true){
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
          if(BinaryOperator *BO = dyn_cast<BinaryOperator>(I)){
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
      std::vector<Value*> RPN(0);
      std::stack<Value*> Stack;
      std::set<Value*> Visited;
      std::set<Value*> PhiNodesVisited;

      errs() << "Getting Expression Tree\n";
      Stack.push(V);

      while (!Stack.empty()) {
        Value *Current = Stack.top();
        Current->dump();
        Stack.pop();
        if(PhiNodesVisited.find(Current) != PhiNodesVisited.end()) {
          RPN.push_back(Current);
          continue;
        }
        if(Visited.find(Current) != Visited.end()) {
          errs() << "hi\n";
          continue;
        }
        RPN.push_back(Current);
        if(TerminalValues.find(Current)!=TerminalValues.end()){
          continue;
        }
        // iterate through operands
        if (isa<Instruction>(Current)) {
          auto *In = dyn_cast<Instruction>(Current);
          if(auto * LI = dyn_cast<LoadInst>(In)) {
            Stack.push(LI->getPointerOperand());
          } else if (auto * SI = dyn_cast<StoreInst>(In)) {
            Stack.push(SI->getPointerOperand());
          } else if (auto * GEPI = dyn_cast<GetElementPtrInst>(In)) {
            for(int i = 1; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
              Stack.push(GEPI->getOperand(i));
            }
          } else if (auto * Phi = dyn_cast<PHINode>(In)) {
            for (auto &Operand : In->operands()) {
              Stack.push(Operand);
            }
            PhiNodesVisited.insert(Phi);
          } else{
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
        if(TerminalValues.find(*RPNIter)!=TerminalValues.end() || isa<ConstantInt> (*RPNIter)){
          errs() << "terminal ";
        }
        else {
          errs() << "operand ";
        }
        (*RPNIter)->dump();
      }
      errs() << "\n";

      return RPN;
    }

    unsigned long long ComputeNumThreads(CallBase* CI) {
      errs() << "computing #threads\n";
      CI->dump();
      // blockdim is assumed to be constant
      unsigned long long bdimx, bdimy, gdimx, gdimy;
      bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
      bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];
      if(KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX] == 0) {
        auto GridSizeValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
        errs() << "grid size x\n";
        GridSizeValue->dump();
        std::vector<Value*> RRPN = getExpressionTree(GridSizeValue);
        gdimx = evaluateRPNForIter0(CI, RRPN);
        /* gdimx = 1; */
      } else {
        gdimx = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX];
      }
      if(KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY] == 0) {
        auto GridSizeValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY];
        errs() << "grid size y\n";
        GridSizeValue->dump();
        std::vector<Value*> RRPN = getExpressionTree(GridSizeValue);
        gdimy = evaluateRPNForIter0(CI, RRPN);
        /* gdimy = 1; */
      } else {
        gdimy = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY];
      }
        errs() << "bdimx = " << bdimx << "\n";
        errs() << "gdimy = " << bdimy << "\n";
        errs() << "gdimx = " << gdimx << "\n";
        errs() << "gdimy = " << gdimy << "\n";
      unsigned long long threads =  bdimx * bdimy * gdimx * gdimy;
      return threads;
    }

    bool isSpecialTypeApp(CallBase* CI) {
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      // TODO: replace with (if memory size unknown)
      if(KernelName.compare("_Z30__device_stub__mummergpuKernelPvPcP12_PixelOfNodeP16_PixelOfChildrenS0_PKiS6_ii") == 0) {
        errs() << "special app\n";
        return true;
      }
      return false;
    }

    void insertCodeToComputeAccessDensity(Function *F, CallBase *CI) {
      errs() << "called function dynamic AD computation\n";
      // to compute access density, we need the following in real time.
      // 1. number of threads
      // 2. loop count inside the kernel
      // 3. size of data structure 
      CI->dump();
      auto GridDimXValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
      GridDimXValue->dump();

      if(auto GridDimX = dyn_cast<Value>(GridDimXValue)){
        errs() << "is here\n";
        LLVMContext &Ctx = F->getContext();
        IRBuilder<> Builder(CI);
        auto printIntFunc = F->getParent()->getOrInsertFunction(
            "print_value_i64",Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
        Value *Args[] = { GridDimX };
        auto PrintFunc = Builder.CreateCall(printIntFunc, Args);
      }

      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      std::map<unsigned, std::vector<std::string>> kernelLoopToBoundsMap = LoopIDToLoopBoundsMap[OriginalKernelName];
      for (auto LoopID = kernelLoopToBoundsMap.begin(); LoopID != kernelLoopToBoundsMap.end(); LoopID++) {
        errs() << "Loop ID = " << LoopID->first << "\n";
        auto LoopIters = partiallyEvaluatedLoopIters(CI, OriginalKernelName, LoopID->first);
        if(LoopIters == nullptr) {
          errs() << "\nPANIC: serious problem with partially evaluated loop iters\n";
        } else {
          std::map<ExprTreeNode*, Value*> Unknowns;
          identifyUnknownsFromExpressionTree(CI, Unknowns, LoopIters);
          errs() << "\nID unknowns\n";
          for(auto unknownIter = Unknowns.begin(); unknownIter != Unknowns.end(); unknownIter++) {
            (*unknownIter).second->dump();
          }
          insertLoopItersEvaluationCode(CI, Unknowns, LoopIters);
        }
      }
      return;
    }

    void identifyUnknownsFromExpressionTree(CallBase* CI, std::map<ExprTreeNode*, Value*> &Unknowns, ExprTreeNode* node) {
      /* std::vector<Value*> unknowns; */
      if(node == nullptr) return;
      /* errs() << "unknown check? " << node->original_str << "\n"; */
      if(isTerminal(node)) {
        /* errs() << "unknown check? is terminal\n" << "\n"; */
        if(node->op == ETO_ARG) {
          /* errs() << "unknown: arg " << node->arg << " \n"; */
          auto unknown = KernelInvocationToArgNumberToActualArgMap[CI][node->arg];
          /* unknown->dump(); */
          Unknowns[node] = (unknown);
        }
      } else {
        /* errs() << "unknown check? not terminal\n" << "\n"; */
        identifyUnknownsFromExpressionTree(CI, Unknowns, node->children[0]);
        identifyUnknownsFromExpressionTree(CI, Unknowns, node->children[1]);
      }
    }

    // TODO: Unknowns is not the correct word for describing what is currently called so.
    Value* insertLoopItersEvaluationCode(CallBase *CI, std::map<ExprTreeNode*, Value*> Unknowns, ExprTreeNode* node) {
      if(node == nullptr) return nullptr;
      if(isTerminal(node)) {
        // handle this node
        errs() << "iliec: " << node->original_str << "\n";
        if(Unknowns.find(node) != Unknowns.end()) {
          Value* val = Unknowns[node];
          val->dump();
          return val;
        }
      } else {
        auto left = insertLoopItersEvaluationCode(CI, Unknowns, node->children[0]);
        auto right = insertLoopItersEvaluationCode(CI, Unknowns, node->children[1]);
        // handle this node
        // create a new LLVM value (Instruction) with the operation in question
        // it will take the left and right children as operands
        // return the new LLVM value
      }
      return nullptr;
    }

    ExprTreeNode* partiallyEvaluatedLoopIters(CallBase *CI, std::string kernelName, int loopID){
      if(loopID == 0) 
        return nullptr;
      std::map<unsigned, std::vector<std::string>> kernelLoopToBoundsMap = LoopIDToLoopBoundsMap[kernelName];
      if(kernelLoopToBoundsMap.find(loopID) != kernelLoopToBoundsMap.end()){
        // compute the loop bound after substituting actual values
        std::vector<std::string> LoopBoundsTokens = kernelLoopToBoundsMap[loopID];
        /* errs() << "\nloop tokens\n"; */
        ExprTreeNode *In, *Fin, *Step;
        unsigned long long InVal, FinVal, StepVal;
        std::vector<std::string> SplitTokens[3];
        unsigned currentSplit = 0;
        for(auto Token = LoopBoundsTokens.begin(); Token != LoopBoundsTokens.end(); Token++) {
          errs() << *Token << " ";
          if((*Token).compare("IN") == 0) {
            currentSplit = 0;
          } else if((*Token).compare("FIN") == 0) {
            currentSplit = 1;
          } else if((*Token).compare("STEP") == 0) {
            currentSplit = 2;
          } else {
            SplitTokens[currentSplit].push_back(*Token);
          }
        }
        In = createExpressionTree(SplitTokens[0]);
        Fin = createExpressionTree(SplitTokens[1]);
        Step = createExpressionTree(SplitTokens[2]);
        // TODO: simplify, combine and return
        return Fin;
      }
      return nullptr;
    }

    // By this point, every data point needed has been already collected.
    void computeAccessDensity(CallBase *CI) {
      // Number of threads * loop iters  / allocation size
      /* unsigned numThreads = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX] */
      /*   * KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY] */
      /*   * KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX] */
      /*   * KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY]; */
      unsigned numThreads = ComputeNumThreads(CI);
      errs() << "# thread = ";
      errs() << numThreads << "\n";

      std::vector<unsigned> numAccessesPerArgNum; // memory allocations are arguements to kernel launch
                                                  //
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      /* errs() << "Name of kernel = " << KernelName << "\n"; */
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      /* errs() << "Original kernel name = " << OriginalKernelName << "\n"; */
      // test
      /* errs() << "loop 1 has " << computeLoopIterations (CI, OriginalKernelName, 1) << " iterations\n"; */

      std::set<unsigned> MemoryAllocArguments;
      std::map<unsigned, unsigned long long > MemoryAllocToNumAccessMap;

      std::map<unsigned, unsigned> AccessIDToAllocMap = AccessIDToAllocationArgMap[OriginalKernelName];
      for (auto AID = AccessIDToAllocMap.begin(); AID != AccessIDToAllocMap.end(); AID++) {
        MemoryAllocArguments.insert(AID->second);
        MemoryAllocToNumAccessMap[AID->second] = 0;
      }
      /* errs() << "\ndebug: size of allocation map = " << MemoryAllocArguments.size() << "\n"; */

      // iterate over all accesses in ADF, the identify their loops and compute total number of accesses 
      std::map<unsigned, unsigned> AccessIDToLoopMap = AccessIDToEnclosingLoopMap[OriginalKernelName];
      for (auto AID = AccessIDToLoopMap.begin(); AID != AccessIDToLoopMap.end(); AID++) {
        errs() << "AID = " << AID->first;
        /* errs() << " " << "Loop is " << AID->second ; */
        unsigned AllocID = AccessIDToAllocationArgMap[OriginalKernelName][AID->first];
        unsigned LoopIters = computeLoopIterations(CI, OriginalKernelName, AID->second);
        errs() << " " << "Iters is " << LoopIters;
        errs() << " " << "Allocation is " <<  AllocID;
        errs() << "num threads = " << numThreads ;
        errs() << "\n";
        // TODO: assert allocID is found in MemoryAllocArguments
        MemoryAllocToNumAccessMap[AllocID] += (unsigned long long) numThreads * LoopIters;
        KernelInvocationToAccessIDToAccessDensity[CI][AID->first] = (unsigned long long) numThreads * LoopIters;
      }

      for (auto AID = MemoryAllocToNumAccessMap.begin(); AID != MemoryAllocToNumAccessMap.end(); AID++) {
        errs() << "\nNum access of " << AID->first << " = " << AID->second << "\n";
        errs() << "size of allocation = " << getAllocationSize(CI,AID->first) << "\n";
        float ad = float(AID->second) / float(getAllocationSize(CI, AID->first));
        errs() << "Access density = " << ad;
      }

      return;
    }

    bool isPointerChase(ExprTreeNode* root) {
      if(root->op == ETO_PC) {
        return true;
      }
      return false;
    }

    void computeMovement(CallBase *CI) {
      errs() << "compute movement\n";
      CI->dump();
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      errs() << "Name of kernel = " << KernelName << "\n";
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      errs() << "Original kernel name = " << OriginalKernelName << "\n";
      std::map<unsigned, ExprTreeNode*> AccessIDToExprMap = AccessIDToExpressionTreeMap[OriginalKernelName];
      auto LIV = KernelInvocationToEnclosingLoopMap[CI];
      unsigned LoopArg = KernelInvocationToLIVToArgNumMap[CI][LIV];
      for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end(); AID++) {
        errs() << "\nAID = " << AID->first;
        traverseExpressionTree(AID->second);
        KernelInvocationToAccessIDToPartDiff_phi[CI][AID->first] = 
          partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_PHI, 0);
        auto pdwrtphi = partialDifferenceWRTPhi(CI, AID->second);
        if(pdwrtphi > 0) {
          KernelInvocationToAccessIDToPartDiff_phi[CI][AID->first] *= pdwrtphi;
        }
        KernelInvocationToAccessIDToPartDiff_bidx[CI][AID->first] = 
          partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_BIDX, 0);
        KernelInvocationToAccessIDToPartDiff_bidy[CI][AID->first] = 
          partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_BIDY, 0);
        KernelInvocationToAccessIDToPartDiff_looparg[CI][AID->first] = 
          partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_ARG, LoopArg);
      }
      errs() << "\n";
    }

    void computeMovementSpecial(CallBase *CI) {
      errs() << "compute movement spl\n";
      CI->dump();
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      errs() << "Name of kernel = " << KernelName << "\n";
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      errs() << "Original kernel name = " << OriginalKernelName << "\n";
      std::map<unsigned, ExprTreeNode*> AccessIDToExprMap = AccessIDToExpressionTreeMap[OriginalKernelName];
      for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end(); AID++) {
        errs() << "\nAID = " << AID->first;
        if(AID->second == nullptr) {
          errs() << "is null\n";
          continue;
        } else {
          traverseExpressionTree(AID->second);
          KernelInvocationToAccessIDToPartDiff_bidx[CI][AID->first] = 
            partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_BIDX, 0);
          KernelInvocationToAccessIDToPartDiff_bidy[CI][AID->first] = 
            partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_BIDY, 0);
          KernelInvocationToAccessIDToPartDiff_phi[CI][AID->first] = 
            partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_PHI, 0);
          auto pdwrtphi = partialDifferenceWRTPhi(CI, AID->second);
          errs() << pdwrtphi << "\n";
          if(pdwrtphi > 0) {
            KernelInvocationToAccessIDToPartDiff_phi[CI][AID->first] *= pdwrtphi;
          }
        }
      }

      return;
    }

    // print for now
    unsigned computeMaxForAccess(CallBase* CI, ExprTreeNode* root, unsigned LoopArg, unsigned loopid) {
      /* errs() << "\ncomputing max Max for " << root->original_str << "\n"; */
      // Convert to RPN
      std::stack<ExprTreeNode*> Stack;
      std::vector<ExprTreeNode*> RPN;
      Stack.push(root);
      while (!Stack.empty()) {
        ExprTreeNode *Current = Stack.top();
        /* errs() << Current->original_str << " "; */
        Stack.pop();
        RPN.push_back(Current);
        if(isOperation(Current)){
          Stack.push(Current->children[0]);
          Stack.push(Current->children[1]);
        }
      }
      reverse(RPN.begin(), RPN.end());
      // Evaluate RPN
      unsigned long long value = evaluateRPNforMax(CI, RPN, LoopArg, loopid);
      errs() << "\n Max = " << value << "\n";
      return value;
    }

    // print for now
    unsigned computeMinForAccess(CallBase* CI, ExprTreeNode* root, unsigned LoopArg, unsigned loopid) {
      /* errs() << "\ncomputing max min for " << root->original_str << "\n"; */
      // Convert to RPN
      std::stack<ExprTreeNode*> Stack;
      std::vector<ExprTreeNode*> RPN;
      Stack.push(root);
      while (!Stack.empty()) {
        ExprTreeNode *Current = Stack.top();
        /* errs() << Current->original_str << " "; */
        Stack.pop();
        RPN.push_back(Current);
        if(isOperation(Current)){
          Stack.push(Current->children[0]);
          Stack.push(Current->children[1]);
        }
      }
      reverse(RPN.begin(), RPN.end());
      // Evaluate RPN
      unsigned long long value = evaluateRPNforMin(CI, RPN, LoopArg, loopid);
      errs() << "\n min = " << value << "\n";
      return value;
    }

    // for each allocation, find the smallest and largest index being accessed in a given iteration
    void computeMinMaxForAllocationForCall(CallBase *CI) {
      errs() << "\ncompute min max for allocation\n";
      CI->dump();
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      /* errs() << "Name of kernel = " << KernelName << "\n"; */
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      /* errs() << "Original kernel name = " << OriginalKernelName << "\n"; */
      std::map<unsigned, ExprTreeNode*> AccessIDToExprMap = AccessIDToExpressionTreeMap[OriginalKernelName];
      auto LIV = KernelInvocationToEnclosingLoopMap[CI];
      unsigned LoopArg = KernelInvocationToLIVToArgNumMap[CI][LIV];
      errs () << "loop arg = " << LoopArg;
      for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end(); AID++) {
        errs() << "\nAID = " << AID->first;
        auto loopid = AccessIDToEnclosingLoopMap[OriginalKernelName][AID->first];
        /* traverseExpressionTree(AID->second); */
        // LoopArg is the kernel argument that is a host loop variable.
        // loopid is the kernel loop id.
        unsigned min = computeMinForAccess(CI, AID->second, LoopArg, loopid);
        unsigned max  = computeMaxForAccess(CI, AID->second, LoopArg, loopid);
        KernelInvocationToAccessIDToWSS[CI][AID->first] = max - min;
        /* partialDifferenceOfExpressionTreeWRTGivenNode(CI, AID->second, ETO_ARG, LoopArg); */
      }
      errs() << "\n";
      return;
    }

    // For given kernel invocation, compute access density, access pattern etc.
    /* void processKernelInvocation( */
    /*     CallBase *CI, std::vector<struct AllocationStruct> &AllocationStructs) { */
    /*   errs() << "Processing Kernel Invoction\n"; */
    /*   CI->dump(); */

    /*   // Identify allocations used in kernel. */
    /*   for (auto Iter = KernelInvocationToArgNumberToAllocationMap[CI].begin(); */
    /*       Iter != KernelInvocationToArgNumberToAllocationMap[CI].end(); Iter++) { */
    /*     errs() << "Arg# = " << Iter->first << "\n"; */
    /*     Iter->second->dump(); */
    /*     auto *OriginalPointer = PointerOpToOriginalPointers[Iter->second]; */
    /*     auto AllocationSize = MallocPointerToSizeMap[OriginalPointer]; */
    /*     errs() << "Size of allocation = " << AllocationSize << "\n"; */
    /*   } */
    /*   for (auto Iter = KernelInvocationToArgNumberToConstantMap[CI].begin(); */
    /*       Iter != KernelInvocationToArgNumberToConstantMap[CI].end(); Iter++) { */
    /*     errs() << "Arg# = " << Iter->first << "\n"; */
    /*     Iter->second->dump(); */
    /*   } */

    /*   // REMOVE */
    /*   /1* return; *1/ */

    /*   // Iterate through accesses in the kernel. */
    /*   auto *KernelPointer = CI->getArgOperand(0); */
    /*   auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer); */
    /*   auto KernelName = KernelFunction->getName(); */
    /*   errs() << "Name of kernel = " << KernelName << "\n"; */
    /*   std::string OriginalKernelName = getOriginalKernelName(KernelName.str()); */
    /*   errs() << "Original kernel name = " << OriginalKernelName << "\n"; */
    /*   std::map<unsigned, unsigned> AccessIDToLoopMap = AccessIDToEnclosingLoopMap[OriginalKernelName]; */
    /*   for (auto AID = AccessIDToLoopMap.begin(); AID != AccessIDToLoopMap.end(); AID++) { */
    /*     errs() << "\nAID = " << AID->first << " loop : " << AID->second; */
    /*   } */
    /*   std::map<unsigned, ExprTreeNode*> AccessIDToExprMap = AccessIDToExpressionTreeMap[OriginalKernelName]; */
    /*   for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end(); AID++) { */
    /*     errs() << "\nAID = " << AID->first; */
    /*     traverseExpressionTree(AID->second); */
    /*   } */
    /*   errs() << "\n"; */

    /* } */

    void processKernelInvocation(CallBase* CI) {
      /* errs() << "enclosing function\n"; */
      KernelInvocationToEnclosingFunction[CI] = CI->getParent()->getParent();
      /* CI->getParent()->getParent()->dump(); */
    }

    // identify if a kernel invocation is inside a loop or not
    bool identifyIterative(CallBase* CI, LoopInfo &LI, ScalarEvolution &SE)  {
      Loop *loop;
      if(loop = LI.getLoopFor(CI->getParent())) {
        errs() << "loop found\n";
        loop->dump();
        auto *LIV = (loop)->getInductionVariable(SE);
        if(LIV){
          errs() << "LIV : ";
          LIV->dump();
          KernelInvocationToEnclosingLoopMap[CI] = LIV;
        }
        auto loopbounds = loop->getBounds(SE);
        if(loopbounds) {
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
        }
        else {
          errs() << "bound not found\n";
        }
        return true;
      }
      return false;
    }

    // For each kernel invocation, collect information related to the invocation
    // and all the memory allocations it is using.
    // The name is not accurate, it collects reuse distance information too.
    void processKernelAccessDensity(
        CallBase *CI, std::vector<struct AllocationStruct> &AllocationStructs) {
      // local containers
      std::vector<Value *> AllocationInsts;
      std::vector<int> NumberOfAccesses;
      std::vector<unsigned long long> AllocationSizes;
      std::vector<std::vector<unsigned>> IndexAxisConstantsLists;

      for (auto Iter = KernelInvocationToArgNumberToAllocationMap[CI].begin();
          Iter != KernelInvocationToArgNumberToAllocationMap[CI].end(); Iter++) {
        errs() << "Arg# = " << Iter->first << "\n";
        Iter->second->dump();
        auto *OriginalPointer = PointerOpToOriginalPointers[Iter->second];
        auto AllocationSize = MallocPointerToSizeMap[OriginalPointer];
        errs() << "Size of allocation = " << AllocationSize << "\n";
        auto *KernelPointer = CI->getOperand(0);
        if (auto *KernelFunction = dyn_cast<Function>(KernelPointer)) {
          auto KernelName = KernelFunction->getName();
          // errs() << KernelName.str() << "\n";
          std::string OriginalKernelName =
            getOriginalKernelName(KernelName.str());
          auto NumAccessInKernel =
            KernelParamUsageInKernel[OriginalKernelName][Iter->first].second;
          errs() << "Number of access in kernel = " << NumAccessInKernel << "\n";
          errs() << "Access Density = "
            << (float)NumAccessInKernel / (float)AllocationSize << "\n";
          AllocationInsts.push_back(Iter->second);
          NumberOfAccesses.push_back(NumAccessInKernel);
          AllocationSizes.push_back(AllocationSize);

          std::vector<unsigned> IndexAxisConstants(INDEX_AXIS_MAX);
          // TODO: very casually using Index Type enum as iteratable
          // FIXME: use a vector instead of map for the list of mulipliers
          for (int IndexType = INDEX_AXIS_LOOPVAR; IndexType != INDEX_AXIS_MAX;
              IndexType++) {
            IndexAxisConstants[IndexType] = 1;
            std::vector<std::string> Multipliers =
              KernelParamReuseInKernel[OriginalKernelName][Iter->first]
              [IndexAxisType(IndexType)];
            if (Multipliers.empty()) {
              continue;
            }
            errs() << IndexAxisTypeToString[IndexAxisType(IndexType)] << " -> ";
            for (auto Multiplier : Multipliers) {
              unsigned ConstMul;
              if (std::atoi(Multiplier.c_str()) == 0) {
                errs() << Multiplier << "=";
                errs() << KernelInvocationToBlockSizeMap
                  [CI][StringToBlockSizeType[Multiplier]]
                  << " ";
                ConstMul = KernelInvocationToBlockSizeMap
                  [CI][StringToBlockSizeType[Multiplier]];
              } else {
                errs() << Multiplier << " ";
                ConstMul = std::atoi(Multiplier.c_str());
              }
              IndexAxisConstants[IndexType] *= ConstMul;
            }
            errs() << "\n";
          }
          IndexAxisConstantsLists.push_back(IndexAxisConstants);
        }
      }

      for (unsigned long I = 0; I < AllocationInsts.size(); I++) {
        auto *Allocation = AllocationInsts[I];
        auto NumberOfAccess = NumberOfAccesses[I];
        auto AllocationSize = AllocationSizes[I];
        auto IndexAxisConstants = IndexAxisConstantsLists[I];
        for (int IndexType = INDEX_AXIS_LOOPVAR; IndexType != INDEX_AXIS_MAX;
            IndexType++) {
          errs() << IndexAxisConstants[IndexType];
        }
        auto AllocationStructObject = AllocationStruct();
        AllocationStructObject.AllocationInst = Allocation;
        AllocationStructObject.AccessCount = NumberOfAccess;
        AllocationStructObject.Size = AllocationSize;
        AllocationStructObject.Density =
          (float)NumberOfAccess / (float)AllocationSize;
        AllocationStructObject.IndexAxisConstants = IndexAxisConstantsLists[I];
        AllocationStructs.push_back(AllocationStructObject);
      }

      std::sort(AllocationStructs.begin(), AllocationStructs.end(),
          allocationSorter);
    }

    void insertAdvisoryCalls(Function *F, CallBase *CI, Value *AllocationInst,
        unsigned long long Size, AdvisoryType Advisory,
        int DeviceId) {

      LLVMContext &Ctx = F->getContext();
      IRBuilder<> Builder(CI);

      if (Advisory == ADVISORY_SET_ACCESSED_BY ||
          Advisory == ADVISORY_SET_PREFERRED_LOCATION) {
        auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
        auto CudaMemAdviseFunc = F->getParent()->getOrInsertFunction(
            "cudaMemAdvise", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
            Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
        unsigned Advise = 0;
        if (Advisory == ADVISORY_SET_PREFERRED_LOCATION) {
          Advise = 3;
        } else if (Advisory == ADVISORY_SET_ACCESSED_BY) {
          Advise = 5;
        }
        Value *Args[] = {
          AllocationInst, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
          ConstantInt::get(Type::getInt32Ty(Ctx), Advise, false),
          ConstantInt::get(Type::getInt32Ty(Ctx), DeviceId, false)};
        auto *SetResident = Builder.CreateCall(CudaMemAdviseFunc, Args);
      } else if (Advisory == ADVISORY_SET_PRIORITIZED_LOCATION) {
        auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
        auto PenguinSetPrioritizedLocation = F->getParent()->getOrInsertFunction(
            "penguinSetPrioritizedLocation", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
            Type::getInt32Ty(Ctx));
        Value *Args[] = {
          AllocationInst, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
          ConstantInt::get(Type::getInt32Ty(Ctx), DeviceId, false)};
        auto *SetResident = Builder.CreateCall(PenguinSetPrioritizedLocation, Args);
      }
      return;
    }

    unsigned long long estimateWorkingSet(unsigned size, unsigned pd_phi, unsigned pd_bidx, unsigned pd_bidy, unsigned loopiters, unsigned gdimx, unsigned gdimy, unsigned bdimx, unsigned bdimy){
      unsigned long long max = 0;
      if((pd_phi * loopiters) > max) {
        max = pd_phi * loopiters;
      }
      unsigned long long numTBs = (1536)/(bdimx * bdimy);
      numTBs = numTBs * 4 * 82;
      unsigned long long concTBx = (gdimx > numTBs) ? numTBs: gdimx;
      unsigned long long concTBy = (gdimx > numTBs) ? 1 : (numTBs/gdimx);
      if((pd_bidx * concTBx) > max) {
        max = pd_bidx * concTBx;
      }
      if((pd_bidy * concTBy) > max) {
        max = pd_bidy * concTBy;
      }
      return max;
    }

    bool computeAndPerformPlacementIterative(Function *F, std::vector<CallBase *> CallInsts, LoopInfo &LI, ScalarEvolution &SE) {
  std::vector<struct AllocationStruct> AllocationStructs;
      AllocationStructs.clear();
      errs() << "performing placement\n";

      std::map<Value*, unsigned long long> AllocationToDensityMap;
      std::map<Value*, std::map<unsigned, unsigned long long>> AllocationToParDiffToDensityMap_phi;
      std::map<Value*, std::map<unsigned, unsigned long long>> AllocationToParDiffToDensityMap_bidx;
      std::map<Value*, std::map<unsigned, unsigned long long>> AllocationToParDiffToDensityMap_bidy;
      std::map<Value*, unsigned> AllocationToParDiffMax_phi;
      std::map<Value*, unsigned> AllocationToParDiffMax_bidx;
      std::map<Value*, unsigned> AllocationToParDiffMax_bidy;
      std::map<Value*, std::map<unsigned long long, unsigned long long >> AllocationToWSSToDensityMap;
      std::map<Value*, unsigned long long> AllocationToEstimatedWSS;
      std::map<Value*, bool> AllocationToPCMap;

      for(auto CIter = CallInsts.begin(); CIter != CallInsts.end(); CIter++){
        CallBase* CI = (*CIter);
        auto *KernelPointer = CI->getArgOperand(0);
        auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
        auto KernelName = KernelFunction->getName();
        std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
        // compute access density of each data structure
        std::map<unsigned, unsigned long long> AIDtoADmap = KernelInvocationToAccessIDToAccessDensity[CI];
        std::map<unsigned, unsigned> AIDtoPDmap_phi = KernelInvocationToAccessIDToPartDiff_phi[CI];
        std::map<unsigned, unsigned> AIDtoPDmap_bidx = KernelInvocationToAccessIDToPartDiff_bidx[CI];
        std::map<unsigned, unsigned> AIDtoPDmap_bidy = KernelInvocationToAccessIDToPartDiff_bidy[CI];
        std::map<unsigned, unsigned> AIDtoAllocArgMap = AccessIDToAllocationArgMap[OriginalKernelName];
        std::map<unsigned, Value*> ArgNumToAllocationMap = KernelInvocationToArgNumberToAllocationMap[CI];

        for(auto AID = AIDtoADmap.begin(); AID != AIDtoADmap.end(); AID++) {
          errs() << AID->first <<" " << AID->second << "\n";
          unsigned allocarg = AIDtoAllocArgMap[AID->first];
          Value* allocation = ArgNumToAllocationMap[allocarg];
          allocation->dump();
          AllocationToDensityMap[allocation] += AID->second;
          unsigned pd_phi = AIDtoPDmap_phi[AID->first];
          unsigned pd_bidx = AIDtoPDmap_bidx[AID->first];
          unsigned pd_bidy = AIDtoPDmap_bidy[AID->first];
          AllocationToParDiffToDensityMap_phi[allocation][pd_phi] += AID->second;
          AllocationToParDiffToDensityMap_bidx[allocation][pd_bidx] += AID->second;
          AllocationToParDiffToDensityMap_bidy[allocation][pd_bidy] += AID->second;

          auto size = getAllocationSize(allocation);
          unsigned loopid = AccessIDToEnclosingLoopMap[OriginalKernelName][AID->first];
          unsigned LoopIters = computeLoopIterations(CI, OriginalKernelName, loopid);

          unsigned gdimx = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX];
          unsigned gdimy = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY];
          unsigned bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
          unsigned bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];

          if(gdimx == 0){
            auto GridSizeValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
            /* GridSizeValue->dump(); */
            std::vector<Value*> RRPN = getExpressionTree(GridSizeValue);
            gdimx = evaluateRPNForIter0(CI, RRPN);
          }
          if(gdimy == 0){
            auto GridSizeValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY];
            /* GridSizeValue->dump(); */
            std::vector<Value*> RRPN = getExpressionTree(GridSizeValue);
            gdimy = evaluateRPNForIter0(CI, RRPN);
          }

          errs() << "pd_phi = " << pd_phi << "\n";
          errs() << "pd_bidx = " << pd_bidx << "\n";
          errs() << "pd_bidy = " << pd_bidy << "\n";
          errs() << "loop iters = " << LoopIters << "\n";
          auto wss = KernelInvocationToAccessIDToWSS[CI][AID->first];
          /* errs() << "wss = " << wss << "\n"; */
          AllocationToWSSToDensityMap[allocation][wss] += AID->second;

        }
      }

      for (auto allocation = AllocationToWSSToDensityMap.begin(); allocation != AllocationToWSSToDensityMap.end(); allocation++){
        std::map<unsigned long long, unsigned long long> WSSToDensityMap = allocation->second;
        unsigned max_density = 0, selected_wss = 0;
        // TODO: use weighted pardiff (weighted by density)
        for(auto wss = WSSToDensityMap.begin(); wss != WSSToDensityMap.end(); wss++) {
            /* errs()  << "pardiff selection : " << pardiff->first << " " <<  pardiff->second <<  "\n"; */
          if(wss->second > max_density) {
            max_density = wss->second;
            selected_wss = wss->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected wss = " << selected_wss << "\n"; */
        AllocationToEstimatedWSS[allocation->first] = selected_wss;
      }

      for (auto allocation = AllocationToParDiffToDensityMap_phi.begin(); allocation != AllocationToParDiffToDensityMap_phi.end(); allocation++){
        std::map<unsigned, unsigned long long> ParDiffToDensityMap_phi = allocation->second;
        unsigned max_density = 0, selected_pardiff = 0;
        for(auto pardiff = ParDiffToDensityMap_phi.begin(); pardiff != ParDiffToDensityMap_phi.end(); pardiff++) {
          if(pardiff->second > max_density) {
            max_density = pardiff->second;
            selected_pardiff = pardiff->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected phi pardiff = " << selected_pardiff << "\n"; */
        AllocationToParDiffMax_phi[allocation->first] = selected_pardiff;
      }

      for (auto allocation = AllocationToParDiffToDensityMap_bidx.begin(); allocation != AllocationToParDiffToDensityMap_bidx.end(); allocation++){
        std::map<unsigned, unsigned long long> ParDiffToDensityMap_bidx = allocation->second;
        unsigned max_density = 0, selected_pardiff = 0;
        for(auto pardiff = ParDiffToDensityMap_bidx.begin(); pardiff != ParDiffToDensityMap_bidx.end(); pardiff++) {
          if(pardiff->second > max_density) {
            max_density = pardiff->second;
            selected_pardiff = pardiff->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected bidx pardiff = " << selected_pardiff << "\n"; */
        AllocationToParDiffMax_bidx[allocation->first] = selected_pardiff;
      }

      for (auto allocation = AllocationToParDiffToDensityMap_bidy.begin(); allocation != AllocationToParDiffToDensityMap_bidy.end(); allocation++){
        std::map<unsigned, unsigned long long> ParDiffToDensityMap_bidy = allocation->second;
        unsigned max_density = 0, selected_pardiff = 0;
        for(auto pardiff = ParDiffToDensityMap_bidy.begin(); pardiff != ParDiffToDensityMap_bidy.end(); pardiff++) {
          if(pardiff->second > max_density) {
            max_density = pardiff->second;
            selected_pardiff = pardiff->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected bidy pardiff = " << selected_pardiff << "\n"; */
        AllocationToParDiffMax_bidy[allocation->first] = selected_pardiff;
      }

      // identify the actions for each allocation
      errs() << "identifying actions for each allocation\n";
      unsigned long long availableMemory = GPU_SIZE;
      std::map<Value*, AllocationAccessPatternType> AllocationToTypeMap;
      for(auto allocation = AllocationToDensityMap.begin(); allocation != AllocationToDensityMap.end(); allocation++) {
        (allocation->first)->dump();
        errs() << allocation->first << "\n";
        errs() << "AD = " << (allocation->second) << "\n";
        errs() << "size = " << getAllocationSize(allocation->first) << "\n";
        errs() << "WSS = " << AllocationToEstimatedWSS[allocation->first] << "\n";
        errs() << "PD_phi = " << AllocationToParDiffMax_phi[allocation->first] << "\n";
        errs() << "PD_bidx = " << AllocationToParDiffMax_bidx[allocation->first] << "\n";
        errs() << "PD_bidy = " << AllocationToParDiffMax_bidy[allocation->first] << "\n";
        auto AllocationStructObject = AllocationStruct();
        AllocationStructObject.AllocationInst = allocation->first;
        AllocationStructObject.AccessCount = allocation->second;
        AllocationStructObject.Size = getAllocationSize(allocation->first);
        AllocationStructObject.wss = AllocationToEstimatedWSS[allocation->first];
        AllocationStructObject.Density = double(allocation->second) / double(AllocationStructObject.wss);
        AllocationStructObject.pd_phi = AllocationToParDiffMax_phi[allocation->first];
        AllocationStructObject.pd_bidx = AllocationToParDiffMax_bidx[allocation->first];
        AllocationStructObject.pd_bidy = AllocationToParDiffMax_bidy[allocation->first];
        AllocationStructs.push_back(AllocationStructObject);
      }

      computeAdvisoryIterative(AllocationStructs, AllocationToWSSToDensityMap);

      placeAdvisoryIterative(F, CallInsts, AllocationStructs, LI, SE);

      return true;
    }

    void computeAdvisoryIterative(std::vector<AllocationStruct>  &AllocationStructs, std::map<Value*, std::map<unsigned long long, unsigned long long>> AllocationToWSSToDensityMap) {
      errs() << "compute advisory iterative \n";
      unsigned long long availableMemory = GPU_SIZE;
      errs() << "availableMemory = " << availableMemory << "\n";
      errs() << AllocationStructs.size() << "\n";
      std::sort(AllocationStructs.begin(), AllocationStructs.end(), allocationSorter);
      unsigned long long sum = 0;
      for(int I = 0; I < AllocationStructs.size(); I++) {
        sum += AllocationStructs[I].Size;
      }
      if(sum < GPU_SIZE) {
        for(int I = 0; I < AllocationStructs.size(); I++) {
          AllocationStructs[I].Advisory = ADVISORY_SET_PREFETCH;
          AllocationStructs[I].AdvisorySize = AllocationStructs[I].Size;
          errs() << "prefetch\n";
        }
        return;
      }
      for(int I = 0; I < AllocationStructs.size(); I++) {
        auto &AllocationStruct = AllocationStructs[I];
        double density = float(AllocationStruct.AccessCount) / float(AllocationStruct.wss);
        errs() << "density is " << density << "\n";
        if(density < 0.5) {
          AllocationStruct.Advisory = ADVISORY_SET_PIN_HOST;
          AllocationStruct.AdvisorySize = AllocationStruct.Size;
          AllocationStruct.AllocationInst->dump();
          errs() << "pin host fully\n";
          auto AllocationStructSubObject = new SubAllocationStruct;
          AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_HOST;
          AllocationStructSubObject->StartIndex = 0;
          AllocationStructSubObject->Size = AllocationStruct.Size;
          AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
          errs() << AllocationStruct.SubAllocations.size() << "\n";
        } else {
          // logic
          if(AllocationStruct.wss < ((MIN_ALLOC_PERC * AllocationStruct.Size)/100)) {
            errs() << "availableMemory before = " << availableMemory << "\n";
            availableMemory -= (AllocationStruct.Size * MIN_ALLOC_PERC )/ 100;
            AllocationStruct.Advisory = ADVISORY_SET_PREFETCH;
            AllocationStruct.AdvisorySize = AllocationStruct.Size;
            AllocationStruct.AllocationInst->dump();
            errs() << "prefetch 1\n";
            errs() << "availableMemory after = " << availableMemory << "\n";
            auto AllocationStructSubObject = new SubAllocationStruct;
            AllocationStructSubObject->Advisory = ADVISORY_SET_PREFETCH;
            AllocationStructSubObject->StartIndex = 0;
            AllocationStructSubObject->Size = AllocationStruct.Size;
            AllocationStructSubObject->PrefetchSize = (AllocationStruct.Size * MIN_ALLOC_PERC )/ 100;
            AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
            errs() << AllocationStruct.SubAllocations.size() << "\n";
          } else if(availableMemory) {
            if(AllocationStruct.Size < availableMemory) {
              auto memUsage = (AllocationStruct.Size < availableMemory) ? AllocationStruct.Size : availableMemory;
              errs() << "memusage = " << memUsage << "\n";
              errs() << "availableMemory before = " << availableMemory << "\n";
              availableMemory -= memUsage;
              AllocationStruct.Advisory = ADVISORY_SET_PIN_DEVICE;
              AllocationStruct.AdvisorySize = memUsage;
              AllocationStruct.AllocationInst->dump();
              errs() << "pin gpu with " << memUsage << "\n";
              errs() << "availableMemory after = " << availableMemory << "\n";
              errs() << AllocationStruct.AdvisorySize << "\n";
              auto AllocationStructSubObject = new SubAllocationStruct;
              AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_DEVICE;
              AllocationStructSubObject->StartIndex = 0;
              AllocationStructSubObject->Size = memUsage;
              AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
              errs() << AllocationStruct.SubAllocations.size() << "\n";
            } else {
              // Only a portion can be placed on the GPU.
              // If the allocation has clustered access pattern across iterations, then prefetch.
              // Else pin a portion.
              auto WSSToDensityMap = AllocationToWSSToDensityMap[AllocationStruct.AllocationInst];
              errs() << "WSS check\n";
              float maxDensityPerWSS = 0.0;
              int selectedWSS = 0;
              for(auto wss = WSSToDensityMap.begin(); wss != WSSToDensityMap.end(); wss++) {
                errs() << "wss = " << wss->first << " " << wss->second << "\n";
                auto densityByWSS = float(wss->second) / float(wss->first);
                errs() << "density per wss = " << densityByWSS  << "\n";
                // TODO: emmmmmm.. replace wss->first condition with wss->second condition
                if((wss->first > 262144) && (maxDensityPerWSS < densityByWSS)) {
                  maxDensityPerWSS = densityByWSS;
                  selectedWSS = wss ->first;
                }
              }
              errs() << "selected wss = " << selectedWSS << " with density " << maxDensityPerWSS << "\n";
              if(selectedWSS < ((MIN_ALLOC_PERC * AllocationStruct.Size)/100)) {
                errs() << "availableMemory before = " << availableMemory << "\n";
                availableMemory -= (AllocationStruct.Size * MIN_ALLOC_PERC )/ 100;
                AllocationStruct.Advisory = ADVISORY_SET_PREFETCH;
                AllocationStruct.AdvisorySize = AllocationStruct.Size;
                AllocationStruct.AllocationInst->dump();
                errs() << "prefetch\n";
                errs() << "availableMemory after = " << availableMemory << "\n";
                auto AllocationStructSubObject = new SubAllocationStruct;
                AllocationStructSubObject->Advisory = ADVISORY_SET_PREFETCH;
                AllocationStructSubObject->StartIndex = 0;
                AllocationStructSubObject->Size = AllocationStruct.Size;
                AllocationStructSubObject->PrefetchSize = (AllocationStruct.Size * MIN_ALLOC_PERC )/ 100;
                AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
                errs() << AllocationStruct.SubAllocations.size() << "\n";
                // host soft pin the rest
                auto AllocationStructSubObject2 = new SubAllocationStruct;
                AllocationStructSubObject2->Advisory = ADVISORY_SET_PIN_HOST;
                AllocationStructSubObject2->StartIndex = 0;
                AllocationStructSubObject2->Size = AllocationStruct.Size;
                /* AllocationStructSubObject->PrefetchSize = (AllocationStruct.Size * MIN_ALLOC_PERC )/ 100; */
                AllocationStruct.SubAllocations.push_back(AllocationStructSubObject2);
                errs() << AllocationStruct.SubAllocations.size() << "\n";
              } else {
                // pin host
                errs() << "host pin due to lack of space and large wss\n";
                AllocationStruct.Advisory = ADVISORY_SET_PIN_HOST;
                AllocationStruct.AdvisorySize = AllocationStruct.Size;
                AllocationStruct.AllocationInst->dump();
                errs() << "pin host fully\n";
                auto AllocationStructSubObject = new SubAllocationStruct;
                AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_HOST;
                AllocationStructSubObject->StartIndex = 0;
                AllocationStructSubObject->Size = AllocationStruct.Size;
                AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
                errs() << AllocationStruct.SubAllocations.size() << "\n";
              }
            }
          }
        }
      }
      if(availableMemory > 0) {
        errs() << "more memory available = " << availableMemory << "\n";
        // divide among allocations decided on prefetching.
        unsigned long long sum = 0;
        for(int I = 0; I < AllocationStructs.size(); I++) {
          auto &AllocationStruct = AllocationStructs[I];
          if(AllocationStruct.Advisory == ADVISORY_SET_PREFETCH) {
            sum += AllocationStruct.wss;
          }
        }
        for(int I = 0; I < AllocationStructs.size(); I++) {
          auto &AllocationStruct = AllocationStructs[I];
          if(AllocationStruct.Advisory == ADVISORY_SET_PREFETCH) {
            errs() << "proportional share";
            auto pshare = double(AllocationStruct.wss) / double(sum);
            errs() << pshare << "\n";
            errs() << "existing memory = ";
            errs() << AllocationStruct.SubAllocations[0]->PrefetchSize << "\n";
            errs() << "allocated memory = ";
            errs() << pshare * availableMemory * 0.9 << "\n";
            AllocationStruct.SubAllocations[0]->PrefetchSize += 
              (unsigned long long) (pshare * double(availableMemory)) * 0.9; // safety factor 0.9
          }
        }
      }
    }

    Value* getNearestValueForMemoryPointer(CallBase *CI, Value* allocInst) {
      // iterate over each argnum, find the aliasing pointer and match with allocInst.
      // if matches, then return KernelInvocationToArgNumberToAllocationMap[argnum]
      auto *KernelArgs = CI->getArgOperand(5); // the 5th argument is the kernel argument struct.
      for (llvm::User *Karg : KernelArgs->users()) {
        errs() << "user: ";
        Karg->dump();
        if(auto allocation = KernelInvocationToKernArgToAllocationMap[CI][Karg]) {
          errs() << "allocation\n";
          allocation->dump();
          auto ArgNumToAllocationMap = KernelInvocationToArgNumberToAllocationMap[CI];
          for (auto argnum = ArgNumToAllocationMap.begin(); argnum != ArgNumToAllocationMap.end(); argnum++) {
            if(argnum->second == allocInst) {
              errs() << "Match = " << argnum->first << "\n";
              auto Inst = KernelInvocationToArgNumberToLastStoreMap[CI][argnum->first];
              Inst->dump();
              if(auto StoreI = dyn_cast<StoreInst>(Inst)){
                auto IndirectPtr = findValueForStoreInstruction(StoreI);
                IndirectPtr->dump();
                auto NearestPtr = findStoreInstOrStackCopyWithGivenValueOperand(IndirectPtr);
                NearestPtr->dump();
                /* NearestPtr->getValueOperand()->dump(); */
                return NearestPtr;
              }
            }
          }
        }
      }
      return nullptr;
    }

    /* void placePrefetchCode(Function *F, IRBuilder<> &Builder, Instruction *insertPoint, Value *LIV, unsigned stepsize, unsigned itersPerPrefetch) { */
      /* return; */
    /* } */

    void placeAdvisoryIterative(Function *F, std::vector<CallBase*> CIs, std::vector<AllocationStruct> &AllocationStructs, LoopInfo &LI, ScalarEvolution &SE) {
      errs() << "place adivsory iterative\n";
      LLVMContext &Ctx = F->getContext();
      Loop *loop = LI.getLoopFor(CIs[0]->getParent());
      auto* LIV = loop->getInductionVariable(SE);
      if (LIV) {
        errs() << "LIV; ";
        LIV->dump();
      }
      auto Iters = KernelInvocationToIterMap[CIs[0]];
      auto Step = KernelInvocationToStepsMap[CIs[0]];
      errs() << "iters = " << Iters << "\n";
      errs() << "step = " << Step << "\n";

      BasicBlock *BB_pred = loop->getLoopPreheader();
      BB_pred->dump();
      Instruction *InsertOutsideLoop = BB_pred->getTerminator();
      errs() << "outside loop\n";
      InsertOutsideLoop->dump();
      BasicBlock *BB_header = loop->getHeader();
      Instruction *InsertInLoop = BB_header->getFirstNonPHI();
      errs() << "inside loop\n";
      InsertInLoop->dump();
      IRBuilder<> Builder(InsertOutsideLoop);
      Builder.SetInsertPoint(InsertOutsideLoop);
      /* BB->dump(); */
      errs() << "num alloc structs = " << AllocationStructs.size() << "\n";
      for(int I = 0; I < AllocationStructs.size(); I++) {
        Value* AllocationInst = AllocationStructs[I].AllocationInst;
        errs() << " allocationnnnn\n";
        AllocationInst->dump();
        auto NearestMemPtr = getNearestValueForMemoryPointer(CIs[0], AllocationInst);
        errs() << " nearest ptr\n";
        NearestMemPtr->dump();
        auto SubAllocations =  AllocationStructs[I].SubAllocations;
        errs() << "num suballoc = " << SubAllocations.size() << "\n";
        for(auto S = 0; S < SubAllocations.size(); S++) {
          AdvisoryType Advisory = SubAllocations[S]->Advisory;
          unsigned long long StartIndex = SubAllocations[S]->StartIndex;
          errs() << "start index = " << StartIndex << "\n";
          unsigned long long Size = SubAllocations[S]->Size;
          errs() << "size = " << Size << "\n";
          if (Advisory == ADVISORY_SET_PIN_DEVICE) {
            errs() << "Device pin\n";
            auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
            auto *I8PTy = Type::getInt8Ty(Ctx);
            auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true));
            Builder.SetInsertPoint(InsertOutsideLoop);
            Value *ActualPtr = Builder.CreateGEP(I8PTy, NearestMemPtr, {start}, "supergep");
            auto PenguinSetPrioritizedLocation = F->getParent()->getOrInsertFunction(
                "penguinSetPrioritizedLocation", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx));
            Value *Args[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0, false)};
            Builder.SetInsertPoint(InsertOutsideLoop);
            auto *SetResident = Builder.CreateCall(PenguinSetPrioritizedLocation, Args);
          }
          if (Advisory == ADVISORY_SET_PREFETCH) {
            errs() << "prefetch\n";
            auto PrefetchSize = SubAllocations[S]->PrefetchSize;
            errs() << "size = " << PrefetchSize << "\n";
            auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
            auto *I8PTy = Type::getInt8Ty(Ctx);
            auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true));
            Builder.SetInsertPoint(InsertInLoop);
            Value *ActualPtr = Builder.CreateGEP(I8PTy, NearestMemPtr, {start}, "supergep");
            auto PenguinSuperPrefetch = F->getParent()->getOrInsertFunction(
                "penguinSuperPrefetch", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx));
            auto NumBatches = AllocationStructs[I].Size / PrefetchSize;
            errs() << "num batches = " << NumBatches << "\n";
            auto iterPerBatch = (Iters / NumBatches) * Step;
            errs() << "iterPerBatch = " << iterPerBatch << "\n";
            Value *Args[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), PrefetchSize, false),
              LIV,
              ConstantInt::get(Type::getInt32Ty(Ctx), iterPerBatch, false),
              ConstantInt::get(Type::getInt64Ty(Ctx), AllocationStructs[I].Size, false)};
            Builder.SetInsertPoint(InsertInLoop);
            auto *SuperPrefetch = Builder.CreateCall(PenguinSuperPrefetch, Args);
          }
          if (Advisory == ADVISORY_SET_PIN_HOST) {
      IRBuilder<> Builder2(InsertOutsideLoop);
            errs() << "Host pin\n";
            auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
            auto *I8PTy = Type::getInt8Ty(Ctx);
            auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true));
            Value *ActualPtr = Builder2.CreateGEP(I8PTy, NearestMemPtr, {start}, "pingu");
            Builder2.SetInsertPoint(InsertOutsideLoop);
            auto CudaMemAdviseFunc = F->getParent()->getOrInsertFunction(
                "cudaMemAdvise", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
            unsigned Advise = 5;
            Value *Args[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), Advise, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0, false)};
            Builder2.SetInsertPoint(InsertOutsideLoop);
            auto *SetResident = Builder2.CreateCall(CudaMemAdviseFunc, Args);
            auto PenguinSetNoMigrateRegion = F->getParent()->getOrInsertFunction(
                "penguinSetNoMigrateRegion", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx), Type::getInt8Ty(Ctx));
            Value *NMGArgs[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0, false), ConstantInt::get(Type::getInt8Ty(Ctx), 0, false)};
            Builder2.SetInsertPoint(InsertOutsideLoop);
            auto *SetNoMigrate = Builder2.CreateCall(PenguinSetNoMigrateRegion, NMGArgs);
          }
          if (Advisory == ADVISORY_SET_DEMAND_MIGRATE) {
            errs() << "demand migrate\n";
            /* auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0); */
            /* auto *I8PTy = Type::getInt8Ty(Ctx); */
            /* auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true)); */
            /* Value *ActualPtr = Builder.CreateGEP(I8PTy, NearestMemPtr, {start}, "supergep"); */
            /* auto CudaMemAdviseFunc = F->getParent()->getOrInsertFunction( */
            /*     "cudaMemAdvise", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx), */
            /*     Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)); */
            /* unsigned Advise = 5; */
            /* Value *Args[] = { */
            /*   ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false), */
            /*   ConstantInt::get(Type::getInt32Ty(Ctx), Advise, false), */
            /*   ConstantInt::get(Type::getInt32Ty(Ctx), 0, false)}; */
            /* auto *SetResident = Builder.CreateCall(CudaMemAdviseFunc, Args); */
          }
        }
      }
      return;
    }

    void placeAdvisorySingleKernel(Function *F, CallBase* CI, std::vector<AllocationStruct> &AllocationStructs) {
      errs() << "place advisory single kernel\n";
      errs() << "num alloc structs = " << AllocationStructs.size() << "\n";
      LLVMContext &Ctx = F->getContext();
      IRBuilder<> Builder(CI);
      for(int I = 0; I < AllocationStructs.size(); I++) {
        Value* AllocationInst = AllocationStructs[I].AllocationInst;
        AllocationInst->dump();
        auto NearestMemPtr = getNearestValueForMemoryPointer(CI, AllocationInst);
        NearestMemPtr->dump();
        auto SubAllocations =  AllocationStructs[I].SubAllocations;
        errs() << "num suballoc = " << SubAllocations.size() << "\n";
        for(auto S = 0; S < SubAllocations.size(); S++) {
          AdvisoryType Advisory = SubAllocations[S]->Advisory;
          unsigned long long StartIndex = SubAllocations[S]->StartIndex;
          errs() << "start index = " << StartIndex << "\n";
          unsigned long long Size = SubAllocations[S]->Size;
          errs() << "size = " << Size << "\n";
          if (Advisory == ADVISORY_SET_PIN_DEVICE) {
            errs() << "Device pin\n";
            auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
            auto *I8PTy = Type::getInt8Ty(Ctx);
            auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true));
            Value *ActualPtr = Builder.CreateGEP(I8PTy, NearestMemPtr, {start}, "supergep");
            auto PenguinSetPrioritizedLocation = F->getParent()->getOrInsertFunction(
                "penguinSetPrioritizedLocation", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx));
            Value *Args[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0, false)};
            auto *SetResident = Builder.CreateCall(PenguinSetPrioritizedLocation, Args);
          }
          if (Advisory == ADVISORY_SET_PIN_HOST) {
            errs() << "Host pin\n";
            auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0);
            auto *I8PTy = Type::getInt8Ty(Ctx);
            auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true));
            Value *ActualPtr = Builder.CreateGEP(I8PTy, NearestMemPtr, {start}, "supergep");
            auto CudaMemAdviseFunc = F->getParent()->getOrInsertFunction(
                "cudaMemAdvise", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx));
            unsigned Advise = 5;
            Value *Args[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), Advise, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0, false)};
            auto *SetResident = Builder.CreateCall(CudaMemAdviseFunc, Args);
            auto PenguinSetNoMigrateRegion = F->getParent()->getOrInsertFunction(
                "penguinSetNoMigrateRegion", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx),
                Type::getInt32Ty(Ctx), Type::getInt8Ty(Ctx));
            Value *NMGArgs[] = {
              ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0, false), ConstantInt::get(Type::getInt8Ty(Ctx), 0, false)};
            auto *SetNoMigrate = Builder.CreateCall(PenguinSetNoMigrateRegion, NMGArgs);
          }
          if (Advisory == ADVISORY_SET_DEMAND_MIGRATE) {
            errs() << "demand migrate\n";
            /* auto *I8PPTy = PointerType::get(Type::getInt8PtrTy(Ctx), 0); */
            /* auto *I8PTy = Type::getInt8Ty(Ctx); */
            /* auto start = ConstantInt::get(Ctx, llvm::APInt(64, StartIndex, true)); */
            /* Value *ActualPtr = Builder.CreateGEP(I8PTy, NearestMemPtr, {start}, "supergep"); */
            /* auto CudaMemAdviseFunc = F->getParent()->getOrInsertFunction( */
            /*     "cudaMemAdvise", Type::getInt32Ty(Ctx), I8PPTy, Type::getInt64Ty(Ctx), */
            /*     Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)); */
            /* unsigned Advise = 5; */
            /* Value *Args[] = { */
            /*   ActualPtr, ConstantInt::get(Type::getInt64Ty(Ctx), Size, false), */
            /*   ConstantInt::get(Type::getInt32Ty(Ctx), Advise, false), */
            /*   ConstantInt::get(Type::getInt32Ty(Ctx), 0, false)}; */
            /* auto *SetResident = Builder.CreateCall(CudaMemAdviseFunc, Args); */
          }
        }
      }
      return;
    }

    void printAdvisorySingleKernel(std::vector<AllocationStruct>  &AllocationStructs) {
      errs() << "printing advisory\n";
      for(int I = 0; I < AllocationStructs.size(); I++) {
        AllocationStructs[I].AllocationInst->dump();
        auto SubAllocations =  AllocationStructs[I].SubAllocations;
        errs() << SubAllocations.size() << "\n";
        for(auto S = 0; S < SubAllocations.size(); S++) {
          errs() << SubAllocations[S]->Size << "\n";
        }
      }
    }

    void computeAdvisorySingleKernel(std::vector<AllocationStruct>  &AllocationStructs) {
      errs() << "compute advisory single kernel\n";
      unsigned long long availableMemory = GPU_SIZE;
      errs() << "availableMemory = " << availableMemory << "\n";
      errs() << AllocationStructs.size() << "\n";
      std::sort(AllocationStructs.begin(), AllocationStructs.end(), allocationSorter);
      unsigned long long sum = 0;
      for(int I = 0; I < AllocationStructs.size(); I++) {
        sum += AllocationStructs[I].Size;
      }
      if(sum < GPU_SIZE) {
        for(int I = 0; I < AllocationStructs.size(); I++) {
          AllocationStructs[I].Advisory = ADVISORY_SET_PREFETCH;
          AllocationStructs[I].AdvisorySize = AllocationStructs[I].Size;
          errs() << "prefetch\n";
        }
        return;
      }
      for(int I = 0; I < AllocationStructs.size(); I++) {
        auto &AllocationStruct = AllocationStructs[I];
        double density = float(AllocationStruct.AccessCount) / float(AllocationStruct.Size);
        errs() << "density is " << density << "\n";
        if(AllocationStruct.isPC) {
          errs() << "this allocation is PC\n";
        }
        if(density < 0.5) {
          AllocationStruct.Advisory = ADVISORY_SET_PIN_HOST;
          AllocationStruct.AdvisorySize = AllocationStruct.Size;
          AllocationStruct.AllocationInst->dump();
          errs() << "pin host fully\n";
          auto AllocationStructSubObject = new SubAllocationStruct;
          AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_HOST;
          AllocationStructSubObject->StartIndex = 0;
          AllocationStructSubObject->Size = AllocationStruct.Size;
          AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
          errs() << AllocationStruct.SubAllocations.size() << "\n";
        } else {
          if(availableMemory > 0) {
            if(AllocationStruct.pd_phi <= 32 && AllocationStruct.pd_bidx <= 32) {
              errs() << "availableMemory before = " << availableMemory << "\n";
              availableMemory -= (AllocationStruct.Size * MIN_ALLOC_PERC )/ 100;
              errs() << "availableMemory after = " << availableMemory << "\n";
              AllocationStruct.Advisory = ADVISORY_SET_DEMAND_MIGRATE;
              AllocationStruct.AdvisorySize = AllocationStruct.Size;
              AllocationStruct.AllocationInst->dump();
              errs() << "migrate on demand\n";
              auto AllocationStructSubObject = new SubAllocationStruct;
              AllocationStructSubObject->Advisory = ADVISORY_SET_DEMAND_MIGRATE;
              AllocationStructSubObject->StartIndex = 0;
              AllocationStructSubObject->Size = AllocationStruct.Size;
              AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
              errs() << AllocationStruct.SubAllocations.size() << "\n";
            } else if(AllocationStruct.pd_phi > 1024 || AllocationStruct.pd_bidx > 1024) {
              if(availableMemory >= 0) {
                auto memUsage = (AllocationStruct.Size < availableMemory) ? AllocationStruct.Size : availableMemory;
                errs() << "memusage = " << memUsage << "\n";
                errs() << "availableMemory before = " << availableMemory << "\n";
                availableMemory -= memUsage;
                errs() << "availableMemory after = " << availableMemory << "\n";
                AllocationStruct.Advisory = ADVISORY_SET_PIN_DEVICE;
                AllocationStruct.AdvisorySize = memUsage;
                AllocationStruct.AllocationInst->dump();
                errs() << "pin gpu with " << memUsage << "\n";
                errs() << AllocationStruct.AdvisorySize << "\n";
                auto AllocationStructSubObject = new SubAllocationStruct;
                AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_DEVICE;
                AllocationStructSubObject->StartIndex = 0;
                AllocationStructSubObject->Size = memUsage;
                AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
                errs() << AllocationStruct.SubAllocations.size() << "\n";
                if(memUsage < AllocationStruct.Size) {
                  /* AllocationStruct.Advisory = ADVISORY_SET_PIN_HOST; */
                  /* AllocationStruct.AdvisorySize = AllocationStruct.Size - memUsage; */
                  /* AllocationStruct.AllocationInst->dump(); */
                  errs() << "pin host\n";
                  errs() << AllocationStruct.AdvisorySize << "\n";
                  auto AllocationStructSubObject = new SubAllocationStruct;
                  AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_HOST;
                  AllocationStructSubObject->StartIndex = memUsage + 1;
                  AllocationStructSubObject->Size = AllocationStruct.Size - memUsage;
                  AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
                  errs() << AllocationStruct.SubAllocations.size() << "\n";
                }
              } else {
                AllocationStruct.Advisory = ADVISORY_SET_PIN_HOST;
                AllocationStruct.AdvisorySize = AllocationStruct.Size;
                AllocationStruct.AllocationInst->dump();
                errs() << "pin host fully\n";
                auto AllocationStructSubObject = new SubAllocationStruct;
                AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_HOST;
                AllocationStructSubObject->StartIndex = 0;
                AllocationStructSubObject->Size = AllocationStruct.Size;
                AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
                errs() << AllocationStruct.SubAllocations.size() << "\n";
              }
            } else {
              // TODO: HANDLE
              errs() << "Case not handled\n";
            }
          } else {
            AllocationStruct.Advisory = ADVISORY_SET_PIN_HOST;
            AllocationStruct.AdvisorySize = AllocationStruct.Size;
            AllocationStruct.AllocationInst->dump();
            errs() << "pin host fully\n";
            errs() << AllocationStruct.AdvisorySize << "\n";
            auto AllocationStructSubObject = new SubAllocationStruct;
            AllocationStructSubObject->Advisory = ADVISORY_SET_PIN_HOST;
            AllocationStructSubObject->StartIndex = 0;
            AllocationStructSubObject->Size = AllocationStruct.Size;
            AllocationStruct.SubAllocations.push_back(AllocationStructSubObject);
            errs() << AllocationStruct.SubAllocations.size() << "\n";
          }
        }
      }
      if(availableMemory > 0) {
        errs() << "more memory available = " << availableMemory << "\n";
      }
    }

    bool computeAndPerformPlacementSingleKernel(Function *F, CallBase *CI) {
  std::vector<struct AllocationStruct> AllocationStructs;
      AllocationStructs.clear();

      errs() << "performing placement\n";
      CI->dump();
      auto *KernelPointer = CI->getArgOperand(0);
      auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
      auto KernelName = KernelFunction->getName();
      std::string OriginalKernelName = getOriginalKernelName(KernelName.str());
      // compute access density of each data structure
      std::map<unsigned, unsigned long long> AIDtoADmap = KernelInvocationToAccessIDToAccessDensity[CI];
      std::map<unsigned, unsigned> AIDtoPDmap_phi = KernelInvocationToAccessIDToPartDiff_phi[CI];
      std::map<unsigned, unsigned> AIDtoPDmap_bidx = KernelInvocationToAccessIDToPartDiff_bidx[CI];
      std::map<unsigned, unsigned> AIDtoPDmap_bidy = KernelInvocationToAccessIDToPartDiff_bidy[CI];
      std::map<unsigned, unsigned> AIDtoAllocArgMap = AccessIDToAllocationArgMap[OriginalKernelName];
      std::map<unsigned, Value*> ArgNumToAllocationMap = KernelInvocationToArgNumberToAllocationMap[CI];

      std::map<Value*, unsigned long long> AllocationToDensityMap;
      std::map<Value*, std::map<unsigned, unsigned long long >> AllocationToParDiffToDensityMap_phi;
      std::map<Value*, std::map<unsigned, unsigned long long>> AllocationToParDiffToDensityMap_bidx;
      std::map<Value*, std::map<unsigned, unsigned long long>> AllocationToParDiffToDensityMap_bidy;
      std::map<Value*, unsigned> AllocationToParDiffMax_phi;
      std::map<Value*, unsigned> AllocationToParDiffMax_bidx;
      std::map<Value*, unsigned> AllocationToParDiffMax_bidy;
      std::map<Value*, std::map<unsigned long long, unsigned long long >> AllocationToWSSToDensityMap;
      std::map<Value*, unsigned long long> AllocationToEstimatedWSS;
      std::map<Value*, bool> AllocationToPCMap;

      for(auto AID = AIDtoADmap.begin(); AID != AIDtoADmap.end(); AID++) {
        errs() << "AID: " << AID->first <<" " << AID->second << "\n";
        unsigned allocarg = AIDtoAllocArgMap[AID->first];
        Value* allocation = ArgNumToAllocationMap[allocarg];
        allocation->dump();
        errs() << allocation << "\n";
        AllocationToDensityMap[allocation] += AID->second;
        unsigned pd_phi = AIDtoPDmap_phi[AID->first];
        unsigned pd_bidx = AIDtoPDmap_bidx[AID->first];
        unsigned pd_bidy = AIDtoPDmap_bidy[AID->first];
        // TODO: Use weighted pardiff instead of this garbage.
        AllocationToParDiffToDensityMap_phi[allocation][pd_phi] += AID->second;
        AllocationToParDiffToDensityMap_bidx[allocation][pd_bidx] += AID->second;
        AllocationToParDiffToDensityMap_bidy[allocation][pd_bidy] += AID->second;

        auto size = getAllocationSize(allocation);
        unsigned loopid = AccessIDToEnclosingLoopMap[OriginalKernelName][AID->first];
        unsigned LoopIters = computeLoopIterations(CI, OriginalKernelName, loopid);

        unsigned gdimx = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMX];
        unsigned gdimy = KernelInvocationToGridSizeMap[CI][AXIS_TYPE_GDIMY];
        unsigned bdimx = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMX];
        unsigned bdimy = KernelInvocationToBlockSizeMap[CI][AXIS_TYPE_BDIMY];

        if(gdimx == 0){
          auto GridSizeValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMX];
          /* GridSizeValue->dump(); */
          std::vector<Value*> RRPN = getExpressionTree(GridSizeValue);
          gdimx = evaluateRPNForIter0(CI, RRPN);
        }
        if(gdimy == 0){
          auto GridSizeValue = KernelInvocationToGridSizeValueMap[CI][AXIS_TYPE_GDIMY];
          /* GridSizeValue->dump(); */
          std::vector<Value*> RRPN = getExpressionTree(GridSizeValue);
          gdimy = evaluateRPNForIter0(CI, RRPN);
        }

          bool isPC = isPointerChase(AccessIDToExpressionTreeMap[OriginalKernelName][AID->first]);
          errs() << "is PC = " << isPC << "\n";
          AllocationToPCMap[allocation] |= isPC;


        errs() << "pd_phi = " << pd_phi << "\n";
        errs() << "pd_bidx = " << pd_bidx << "\n";
        errs() << "pd_bidy = " << pd_bidy << "\n";
        errs() << "loop iters = " << LoopIters << "\n";
        auto wss = estimateWorkingSet(size, pd_phi, pd_bidx, pd_bidy, LoopIters, gdimx, gdimy, bdimx, bdimy);
        /* errs() << "wss = " << wss << "\n"; */
        AllocationToWSSToDensityMap[allocation][wss] += AID->second;
      }

      for (auto allocation = AllocationToWSSToDensityMap.begin(); allocation != AllocationToWSSToDensityMap.end(); allocation++){
        std::map<unsigned long long, unsigned long long> WSSToDensityMap = allocation->second;
        unsigned max_density = 0, selected_wss = 0;
        // TODO: use weighted pardiff (weighted by density)
        for(auto wss = WSSToDensityMap.begin(); wss != WSSToDensityMap.end(); wss++) {
            /* errs()  << "pardiff selection : " << pardiff->first << " " <<  pardiff->second <<  "\n"; */
          if(wss->second > max_density) {
            max_density = wss->second;
            selected_wss = wss->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected wss = " << selected_wss << "\n"; */
        AllocationToEstimatedWSS[allocation->first] = selected_wss;
      }

      for (auto allocation = AllocationToParDiffToDensityMap_phi.begin(); allocation != AllocationToParDiffToDensityMap_phi.end(); allocation++){
        std::map<unsigned, unsigned long long> ParDiffToDensityMap_phi = allocation->second;
        unsigned max_density = 0, selected_pardiff = 0;
        // TODO: use weighted pardiff (weighted by density)
        for(auto pardiff = ParDiffToDensityMap_phi.begin(); pardiff != ParDiffToDensityMap_phi.end(); pardiff++) {
            /* errs()  << "pardiff selection : " << pardiff->first << " " <<  pardiff->second <<  "\n"; */
          if(pardiff->second > max_density) {
            max_density = pardiff->second;
            selected_pardiff = pardiff->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected phi pardiff = " << selected_pardiff << "\n"; */
        AllocationToParDiffMax_phi[allocation->first] = selected_pardiff;
      }

      for (auto allocation = AllocationToParDiffToDensityMap_bidx.begin(); allocation != AllocationToParDiffToDensityMap_bidx.end(); allocation++){
        std::map<unsigned, unsigned long long> ParDiffToDensityMap_bidx = allocation->second;
        unsigned max_density = 0, selected_pardiff = 0;
        for(auto pardiff = ParDiffToDensityMap_bidx.begin(); pardiff != ParDiffToDensityMap_bidx.end(); pardiff++) {
          if(pardiff->second > max_density) {
            max_density = pardiff->second;
            selected_pardiff = pardiff->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected bidx pardiff = " << selected_pardiff << "\n"; */
        AllocationToParDiffMax_bidx[allocation->first] = selected_pardiff;
      }

      for (auto allocation = AllocationToParDiffToDensityMap_bidy.begin(); allocation != AllocationToParDiffToDensityMap_bidy.end(); allocation++){
        std::map<unsigned, unsigned long long> ParDiffToDensityMap_bidy = allocation->second;
        unsigned max_density = 0, selected_pardiff = 0;
        for(auto pardiff = ParDiffToDensityMap_bidy.begin(); pardiff != ParDiffToDensityMap_bidy.end(); pardiff++) {
          if(pardiff->second > max_density) {
            max_density = pardiff->second;
            selected_pardiff = pardiff->first;
          }
        }
        (allocation->first)->dump();
        /* errs() << "selected bidy pardiff = " << selected_pardiff << "\n"; */
        AllocationToParDiffMax_bidy[allocation->first] = selected_pardiff;
      }

      for(auto allocation = AllocationToDensityMap.begin(); allocation != AllocationToDensityMap.end(); allocation++) {
        (allocation->first)->dump();
        auto size = getAllocationSize(allocation->first);
        errs() << size << "\n";
        errs() << (double(allocation->second) / double(size)) << "\n";
        unsigned ph_phi = AllocationToParDiffMax_phi[allocation->first];
        unsigned ph_bidx = AllocationToParDiffMax_bidx[allocation->first];
        unsigned ph_bidy = AllocationToParDiffMax_bidy[allocation->first];
      }

      // identify the actions for each allocation
      errs() << "identifying actions for each allocation\n";
      unsigned long long availableMemory = GPU_SIZE;
      std::map<Value*, AllocationAccessPatternType> AllocationToTypeMap;
      for(auto allocation = AllocationToDensityMap.begin(); allocation != AllocationToDensityMap.end(); allocation++) {
        (allocation->first)->dump();
        errs() << allocation->first << "\n";
        errs() << "AD = " << (allocation->second) << "\n";
        errs() << "size = " << getAllocationSize(allocation->first) << "\n";
        errs() << "WSS = " << AllocationToEstimatedWSS[allocation->first] << "\n";
        errs() << "PD_phi = " << AllocationToParDiffMax_phi[allocation->first] << "\n";
        errs() << "PD_bidx = " << AllocationToParDiffMax_bidx[allocation->first] << "\n";
        errs() << "PD_bidy = " << AllocationToParDiffMax_bidy[allocation->first] << "\n";
        auto AllocationStructObject = AllocationStruct();
        AllocationStructObject.AllocationInst = allocation->first;
        AllocationStructObject.AccessCount = allocation->second;
        AllocationStructObject.Size = getAllocationSize(allocation->first);
        AllocationStructObject.Density = (float(AllocationStructObject.AccessCount) / float(AllocationStructObject.Size));
        AllocationStructObject.wss = AllocationToEstimatedWSS[allocation->first];
        AllocationStructObject.pd_phi = AllocationToParDiffMax_phi[allocation->first];
        AllocationStructObject.pd_bidx = AllocationToParDiffMax_bidx[allocation->first];
        AllocationStructObject.pd_bidy = AllocationToParDiffMax_bidy[allocation->first];
        AllocationStructObject.isPC = AllocationToPCMap[allocation->first];
        AllocationStructs.push_back(AllocationStructObject);
      }

      computeAdvisorySingleKernel(AllocationStructs);
      /* printAdvisorySingleKernel(AllocationStructs); */
      placeAdvisorySingleKernel(F, CI, AllocationStructs);

      return true;
    }

        /* std::vector<struct AllocationStruct> AllocationStructs) { */

      /* errs() << "Compute and perfom placement\n"; */

      /* for (unsigned long I = 0; I < AllocationStructs.size(); I++) { */
        /* AllocationStructs[I].AllocationInst->dump(); */
        /* errs() << "density = " << AllocationStructs[I].Density << "\n"; */
        /* errs() << "Index Axis Consts = "; */
        /* for (int IndexType = INDEX_AXIS_LOOPVAR; IndexType != INDEX_AXIS_MAX; */
        /*     IndexType++) { */
        /*   errs() << AllocationStructs[I].IndexAxisConstants[IndexType] << " "; */
        /* } */
        /* errs() << "\n"; */
      /* } */

      /* unsigned long long Used = 0; */

      /* for (unsigned long I = 0; I < AllocationStructs.size(); I++) { */
        /* IRBuilder<> Builder(CI); */
        /* // auto *Pointer = PointerType::get(Type::getInt8Ty(Ctx), 0); */
        /* errs() << "Pre placement: GPU_SIZE = " << GPU_SIZE */
        /*   << " and USED = " << Used */
        /*   << " and alloc size = " << AllocationStructs[I].Size << "\n"; */
        /* if ((Used < GPU_SIZE) && (Used + AllocationStructs[I].Size) <= GPU_SIZE) { */
        /*   Used += AllocationStructs[I].Size; */
        /*   errs() << "USED = " << Used << "\n"; */
        /*   insertAdvisoryCalls(F, CI, AllocationStructs[I].AllocationInst, */
        /*       AllocationStructs[I].Size, */
        /*       ADVISORY_SET_PREFERRED_LOCATION, 0); */
        /* } else if ((Used < GPU_SIZE) && */
        /*     ((Used + AllocationStructs[I].Size) >= GPU_SIZE)) { */
        /*   auto PartialMemoryUsed = GPU_SIZE - Used; */
        /*   Used += AllocationStructs[I].Size; */
        /*   errs() << "USED = " << Used << "\n"; */
        /*   errs() << "Partial memory = " << PartialMemoryUsed << "\n"; */
        /*   insertAdvisoryCalls(F, CI, AllocationStructs[I].AllocationInst, */
        /*       PartialMemoryUsed, */
        /*       ADVISORY_SET_PREFERRED_LOCATION, 0); */
        /*   insertAdvisoryCalls(F, CI, AllocationStructs[I].AllocationInst, */
        /*       AllocationStructs[I].Size, */
        /*       ADVISORY_SET_ACCESSED_BY, 0); */
        /* } else { */
        /*   Used += AllocationStructs[I].Size; */
        /*   errs() << "USED = " << Used << "\n"; */
        /*   insertAdvisoryCalls(F, CI, AllocationStructs[I].AllocationInst, */
        /*       AllocationStructs[I].Size, */
        /*       ADVISORY_SET_PREFERRED_LOCATION, -1); */
        /* } */
      /* } */

      /* return true; */
    /* } */

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
              FunctionCallToActualArgumentToFormalArgumentMap[CallSiteIter->first][ActualArg] = FormalArg;
              /* ActualArgumentToFormalArgumentMap[ActualArg] = (FormalArg); */
              CallSiteIter->first->dump();
              errs() << "formal arg to actual arg\n";
              errs() << FormalArg << "\n";
              FormalArg->dump();
              errs() << ActualArg << "\n";
              ActualArg->dump();
              FunctionCallToFormalArgumentToActualArgumentMap[CallSiteIter->first][FormalArg] = (ActualArg);
            }
          }
        }
        if (MatchCount > 1) {
          errs() << "MORE THAN ONE CALL SITE \n";
        }
      }
    }

    void analyzePointerPropogationRecursive(CallBase *CI) {
      if(VisitedCallInstForPointerPropogation.find(CI) != VisitedCallInstForPointerPropogation.end()) {
        return;
      } else {
        VisitedCallInstForPointerPropogation.insert(CI);
      }
      auto *Func = CI->getCalledFunction();
      errs() << "function name = " << Func->getName() << "\n";
      /* Func->dump(); */
      if(ListOfLocallyDefinedFunctions.find(Func) == ListOfLocallyDefinedFunctions.end()){
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
              // OriginalPointers.insert(&I);
              PointerOpToOriginalPointers[&I] = &I;
            }
            if(auto Stype = dyn_cast<StructType>(AI->getAllocatedType())){
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
              if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                PointerOpToOriginalStructPointer[LI] = POGP->second;
                PointerOpToOriginalStructPointersIndex[LI] = 
                  PointerOpToOriginalStructPointersIndex[LI->getPointerOperand()];
                errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[LI->getPointerOperand()];
              }
            }
          }
          if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
            // LI->getPointerOperand()->dump();
            errs() << "GEPI testing: ";
            GEPI->dump();
            GEPI->getPointerOperand()->dump();
            errs() << GEPI->getPointerOperand() << "\n";
            auto POGP = PointerOpToOriginalPointers.find(GEPI->getPointerOperand());
            if (POGP != PointerOpToOriginalPointers.end()) {
              PointerOpToOriginalPointers[GEPI] = POGP->second;
              errs() << "\nGEPI INST \n";
              GEPI->dump();
              GEPI->getPointerOperand()->dump();
              POGP->second->dump();
              if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                PointerOpToOriginalStructPointer[GEPI] = POGP->second;
                auto numIndices = GEPI->getNumIndices();
                if(numIndices == 2) {
                  if(auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(2))){
                    errs() << "og is struct\n";
                    PointerOpToOriginalStructPointersIndex[GEPI] = FieldNum->getSExtValue();
                    errs() << "field num = " << FieldNum << "\n";
                  }
                } else {
                  if(auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(1))){
                    errs() << "og maybe struct or array\n";
                    PointerOpToOriginalStructPointersIndex[GEPI] = FieldNum->getSExtValue();
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
              if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                PointerOpToOriginalStructPointer[SI->getPointerOperand()] = POGP->second;
                PointerOpToOriginalStructPointersIndex[SI->getPointerOperand()] = 
                  PointerOpToOriginalStructPointersIndex[SI->getValueOperand()];
                errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[SI->getValueOperand()];
              }
            }
          }
          if (auto *CI = dyn_cast<CallBase>(&I)) {
            auto *Callee = CI->getCalledFunction();
            if ((Callee && ((Callee->getName() == "llvm.lifetime.start.p0") ||
                    Callee->getName() == "llvm.lifetime.end.p0"))) {
              continue;
            }
            if ((Callee && (Callee->getName() == "llvm.memcpy.p0.p0.i64")) ){
              errs() << "memcpy found\n";
              CI->getOperand(0)->dump();
              CI->getOperand(1)->dump();
              bool isStackVar0 = isa<AllocaInst>(CI->getOperand(0));
              bool isStackVar1 = isa<AllocaInst>(CI->getOperand(1));
              if(isStackVar0 || isStackVar1) {
                MemcpyOpForStructs.insert(CI);
                MemcpyOpForStructsSrcToInstMap[CI->getOperand(1)] = CI;
                MemcpyOpForStructsDstToInstMap[CI->getOperand(0)] = CI;
                if(PointerOpToOriginalPointers.find(CI->getOperand(1))  != PointerOpToOriginalPointers.end()){
                  errs() << "memcpy taint propogated\n";
                  PointerOpToOriginalPointers[CI->getOperand(1)]->dump();
                  PointerOpToOriginalPointers[CI->getOperand(0)] = PointerOpToOriginalPointers[CI->getOperand(1)];
                  auto OriginalPointer = PointerOpToOriginalPointers[CI->getOperand(1)];
                  if(StructAllocas.find(OriginalPointer) != StructAllocas.end()){
                    PointerOpToOriginalStructPointer[CI->getOperand(0)] = OriginalPointer;
                    PointerOpToOriginalStructPointersIndex[CI->getOperand(0)] = 
                      PointerOpToOriginalStructPointersIndex[CI->getOperand(1)];
                    errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[CI->getOperand(1)];
                  }
                }
              }
            }
            errs() << "CallBase : ";
            CI->dump();
            auto *Func = CI->getCalledFunction();
            if(ListOfLocallyDefinedFunctions.find(Func) == ListOfLocallyDefinedFunctions.end()){
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
                auto ActualArgumentToFormalArgumentMap = FunctionCallToActualArgumentToFormalArgumentMap[CI];
                auto *FormalArg =
                  ActualArgumentToFormalArgumentMap[Arg]; // ->dump();
                FormalArg->dump();
                errs() << FormalArg << "\n";
                PointerOpToOriginalPointers[FormalArg] = OriginalPointer;
                if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                  PointerOpToOriginalStructPointer[FormalArg] = POGP->second;
                  PointerOpToOriginalStructPointersIndex[FormalArg] = 
                    PointerOpToOriginalStructPointersIndex[Arg];
                  errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[Arg];
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
              if(auto Stype = dyn_cast<StructType>(AI->getAllocatedType())){
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
                if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                  PointerOpToOriginalStructPointer[LI] = POGP->second;
                  PointerOpToOriginalStructPointersIndex[LI] = 
                    PointerOpToOriginalStructPointersIndex[LI->getPointerOperand()];
                      errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[LI->getPointerOperand()];
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
                if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                  PointerOpToOriginalStructPointer[SI->getPointerOperand()] = POGP->second;
                  PointerOpToOriginalStructPointersIndex[SI->getPointerOperand()] = 
                    PointerOpToOriginalStructPointersIndex[SI->getValueOperand()];
                      errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[SI->getValueOperand()];
                }
              }
              if(isa<ConstantInt>(SI->getValueOperand())){
                errs() << "Constant store\n";
                auto con = dyn_cast<ConstantInt>(SI->getValueOperand());
                PointerOpToOriginalConstant[SI->getPointerOperand()] = con->getSExtValue();
              }
            }
            if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
              // LI->getPointerOperand()->dump();
              auto POGP = PointerOpToOriginalPointers.find(GEPI->getPointerOperand());
              if (POGP != PointerOpToOriginalPointers.end()) {
                PointerOpToOriginalPointers[GEPI] = POGP->second;
                errs() << "\nGEPI INST \n";
                GEPI->dump();
                GEPI->getPointerOperand()->dump();
                POGP->second->dump();
                if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                  PointerOpToOriginalStructPointer[GEPI] = POGP->second;
                  auto numIndices = GEPI->getNumIndices();
                  if(numIndices == 2) {
                    if(auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(2))){
                      errs() << "og is struct\n";
                      PointerOpToOriginalStructPointersIndex[GEPI] = FieldNum->getSExtValue();
                      errs() << "field num = " << FieldNum << "\n";
                    }
                  } else {
                    if(auto FieldNum = dyn_cast<ConstantInt>(GEPI->getOperand(1))){
                      errs() << "og maybe struct or array\n";
                      PointerOpToOriginalStructPointersIndex[GEPI] = FieldNum->getSExtValue();
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
              if ((Callee && (Callee->getName() == "llvm.memcpy.p0.p0.i64")) ){
                errs() << "memcpy found\n";
                CI->getOperand(0)->dump();
                CI->getOperand(1)->dump();
                bool isStackVar0 = isa<AllocaInst>(CI->getOperand(0));
                bool isStackVar1 = isa<AllocaInst>(CI->getOperand(1));
                if(isStackVar0 || isStackVar1) {
                  MemcpyOpForStructs.insert(CI);
                  MemcpyOpForStructsSrcToInstMap[CI->getOperand(1)] = CI;
                  MemcpyOpForStructsDstToInstMap[CI->getOperand(0)] = CI;
                  if(PointerOpToOriginalPointers.find(CI->getOperand(1))  != PointerOpToOriginalPointers.end()){
                    errs() << "memcpy taint propogated \n ";
                    PointerOpToOriginalPointers[CI->getOperand(1)]->dump();
                    PointerOpToOriginalPointers[CI->getOperand(0)] = PointerOpToOriginalPointers[CI->getOperand(1)];
                    auto OriginalPointer = PointerOpToOriginalPointers[CI->getOperand(1)];
                    if(StructAllocas.find(OriginalPointer) != StructAllocas.end()){
                      PointerOpToOriginalStructPointer[CI->getOperand(0)] = OriginalPointer;
                      PointerOpToOriginalStructPointersIndex[CI->getOperand(0)] = 
                        PointerOpToOriginalStructPointersIndex[CI->getOperand(1)];
                      errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[CI->getOperand(1)];
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
                  auto ActualArgumentToFormalArgumentMap = FunctionCallToActualArgumentToFormalArgumentMap[CI];
                  Value *FormalArg =
                    ActualArgumentToFormalArgumentMap[Arg]; // ->dump();
                  FormalArg->dump();
                  errs() << FormalArg << "\n";
                  PointerOpToOriginalPointers[FormalArg] = OriginalPointer;
                  if(StructAllocas.find(POGP->second) != StructAllocas.end()){
                    PointerOpToOriginalStructPointer[FormalArg] = POGP->second;
                      PointerOpToOriginalStructPointersIndex[FormalArg] = 
                        PointerOpToOriginalStructPointersIndex[Arg];
                      errs() << "zoo zoo = " << PointerOpToOriginalStructPointersIndex[Arg];
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

    void extractArgsFromCall(CallBase *CI) {
      errs() << "CALL \n";
      CI->dump();
      errs() << "NAME \n";
      auto *KernelPointer = CI->getArgOperand(0);
      if (auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer)) {
        auto KernelName = KernelFunction->getName();
        errs() << getOriginalKernelName(KernelName.str()) << "\n";
      }
      errs() << "ARG STRUCT \n";
      auto *KernelArgs =
        CI->getArgOperand(5); // the 5th argument is the kernel argument struct.
      KernelArgs->dump();
      KernelInvocationToStructMap[CI] = KernelArgs;
      // errs() << "USERS \n";
      // for (llvm::User *Karg : KernelArgs->users()) {
      //   Karg->dump();
      // recurseTillStoreOrEmtpy(CI, KernelArgs, Karg);
      // }
      // for (auto &Arg: I->args()){
      //   Arg->dump();
      // }
      return;
    }

    void setTerminals() {
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

      operations.insert(ETO_ADD);
      operations.insert(ETO_AND);
      operations.insert(ETO_OR);
      operations.insert(ETO_MUL);
      operations.insert(ETO_SHL);
      operations.insert(ETO_PHI);
    }

    void printLoopInformation() {
      errs() << "loop information\n";
      for (auto I = LoopIDToLoopItersMap.begin(); I != LoopIDToLoopItersMap.end(); I++) {
        errs() << I->first << "\n";
        for(auto L = I->second.begin(); L != I->second.end(); L++){
          errs() << L->first << " ==> " << L->second << "\n";
        }
      }
      for (auto I = LoopIDToLoopBoundsMap.begin(); I != LoopIDToLoopBoundsMap.end(); I++) {
        errs() << I->first << "\n";
        for(auto L = I->second.begin(); L != I->second.end(); L++){
          errs() << L->first << " ==> ";
          for (auto str = L->second.begin(); str != L->second.end(); str++) {
            errs() << *str << " ";
          }
        }
      }
      errs() << "\n";
    }

    void printAccessInformation() {
      errs() << "access information\n";
      errs() << "\n";
      // the key for all the AccessId maps is the same (kernel name), so use any to iterate
      for (auto I = AccessIDToAllocationArgMap.begin(); I != AccessIDToAllocationArgMap.end(); I++) {
        errs() << "\nkernel name: " << I->first << "\n";
        std::map<unsigned, unsigned> AccessIDToArgMap = AccessIDToAllocationArgMap[I->first];
        errs() << "AID to arg map\n";
        for (auto AID = AccessIDToArgMap.begin(); AID != AccessIDToArgMap.end(); AID++) {
          errs() << AID->first << " " << AID->second << "\n";
        }
        errs() << "AID to loop map\n";
        std::map<unsigned, unsigned> AccessIDToLoopMap = AccessIDToEnclosingLoopMap[I->first];
        for (auto AID = AccessIDToLoopMap.begin(); AID != AccessIDToLoopMap.end(); AID++) {
          errs() << AID->first << " " << AID->second << "\n";
        }
        errs() << "AID to expression tree map\n";
        std::map<unsigned, ExprTreeNode*> AccessIDToExprMap = AccessIDToExpressionTreeMap[I->first];
        for (auto AID = AccessIDToExprMap.begin(); AID != AccessIDToExprMap.end(); AID++) {
          errs() << "\nAID = " << AID->first;
          traverseExpressionTree(AID->second);
        }
      }
    }

    bool doInitialization(Module &M) override {
      setTerminals();
      printKernelDeviceAnalyis();

      printLoopInformation();
      printAccessInformation();

      /* errs() << "KerneL Param Usage In Kernel \n"; */
      /* for (auto I = KernelParamUsageInKernel.begin(); */
      /*     I != KernelParamUsageInKernel.end(); I++) { */
      /*   errs() << I->first << "\n"; */
      /*   for (auto U = (I->second).begin(); U != (I->second).end(); U++) { */
      /*     errs() << U->first << " " << U->second << "\n"; */
      /*   } */
      /* } */
      /* errs() << "Kernel Reuse Usage In Kernel \n"; */
      /* for (auto I = KernelParamReuseInKernel.begin(); */
      /*     I != KernelParamReuseInKernel.end(); I++) { */
      /*   errs() << I->first << "\n"; */
      /*   for (auto U = (I->second).begin(); U != (I->second).end(); U++) { */
      /*     errs() << ".." << U->first << "\n"; */
      /*     for (auto V = (U->second).begin(); V != (U->second).end(); V++) { */
      /*       errs() << "...." << IndexAxisTypeToString[V->first] << "\n"; */
      /*       for (auto W = (V->second).begin(); W != (V->second).end(); W++) { */
      /*         errs() << "......" << *W << "\n"; */
      /*       } */
      /*       errs() << "\n"; */
      /*     } */
      /*   } */
      /* } */

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

      /* return true; */
      for (auto &F : M) {
        if (F.getName().contains("stub")) {
          errs() << "not running on " << F.getName() << "\n";
          continue;
        }
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallBase>(&I)) {
              auto *Callee = CI->getCalledFunction();
              /* errs() << Callee->getName() << "\n"; */
              if (Callee && Callee->getName() == "cudaMallocManaged") {
                processMemoryAllocation(CI);
              }
              if (Callee && Callee->getName() == ("cudaLaunchKernel")) {
                // processKernelSignature(CI);
                // processKernelArguments(CI);
                // extractArgsFromCall(CI);
              }
            }
          }
          // runOnFunctionImpl(F);
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
        if (F.getName().contains("__cuda_module_ctor") || F.getName().contains("__cuda_register_globals")) {
          errs() << "CTOR FOUND\n";
          for (auto &BB : F) {
            for (auto &I : BB) {
              if (auto *CI = dyn_cast<CallBase>(&I)) {
                CI->dump();
                auto *Callee = CI->getCalledFunction();
                if (Callee && Callee->getName() == ("__cudaRegisterFunction")) {
                  errs() <<"Found a registration\n";
                  errs() << Callee->getName() << "\n";
                  /* CI->getArgOperand(1)->dump(); */
                  if (llvm::Function *Fn = dyn_cast<Function>(CI->getArgOperand(1))) {
                    errs() << " func name = " << Fn->getName() << "\n";
                    if (llvm::GlobalVariable* DvFn = dyn_cast<GlobalVariable>(CI->getArgOperand(2))) {
                      errs() << " device side name = ";
                      auto DvFnStr = dyn_cast<ConstantDataArray>(DvFn->getInitializer());
                      if(DvFnStr) {
                        errs() << DvFnStr->getAsCString() << "\n";
                        HostSideKernelNameToOriginalNameMap[std::string(Fn->getName())] = std::string(DvFnStr->getAsCString());
                      }
                    }
                  }
                  }
                }
              }
            }
          }
          else {
            continue;
          }
        }

        bool isIterative = false;
        std::vector<CallBase*> CallInsts;
        std::set<Function*> FunctionsWithKernelLaunches;

        for (auto &F : M) {

          if (F.getName().contains("stub")) {
            errs() << "not running on " << F.getName() << "\n";
            continue;
          }
          for (auto &BB : F) {

            LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
            ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
            for (auto &I : BB) {
              if (auto *CI = dyn_cast<CallBase>(&I)) {
                auto *Callee = CI->getCalledFunction();
                if (Callee && Callee->getName() == ("cudaLaunchKernel")) {

                  bool iterative = identifyIterative(CI, LI, SE);
                  isIterative = isIterative | iterative; 
                  CallInsts.push_back(CI);

                  if(iterative) {
                    errs() << "launch is iterative\n";
                  } else {
                    errs() << "launch is NOT iterative\n";
                  }

                  auto *KernelPointer = CI->getArgOperand(0);
                  auto *KernelFunction = dyn_cast_or_null<Function>(KernelPointer);
                  auto KernelName = KernelFunction->getName();
                  errs() << "Name of kernel = " << KernelName << "\n";
                  if(KernelName.compare("_ZN8GpuBTree7kernels25__device_stub__init_btreeI13PoolAllocatorEEvPjT_") == 0) continue;
                  if(KernelName.compare("_ZN8GpuBTree7kernels26__device_stub__insert_keysIjjj13PoolAllocatorEEvPjPT_PT0_T1_T2_") == 0) continue;
                  if(KernelName.compare("_Z32__device_stub__mummergpuRCKernelP10MatchCoordPcPKiS3_ii") == 0) continue;
                  if(KernelName.compare("_Z26__device_stub__printKernelP9MatchInfoiP9AlignmentPcP12_PixelOfNodeP16_PixelOfChildrenPKiS9_iiiii") == 0) continue;

                  processKernelInvocation(CI);
                  processKernelSignature(CI);
                  processKernelArguments(CI);

                  FunctionsWithKernelLaunches.insert(&F);

                  insertCodeToComputeAccessDensity(&F, CI);

                  /* computeAccessDensity(CI); */
                  /* computeMovement(CI); */

                  /* if(iterative){ */
                  /*   computeMinMaxForAllocationForCall(CI); */
                  /* } else { */
                  /*   if(!isIterative){ */
                  /*     errs() << "not skipping\n"; */
                  /*     computeAndPerformPlacementSingleKernel(&F, CI); */
                  /*   } else { */
                  /*     errs() << "skipping\n"; */
                  /*   } */
                  /* } */

                }
              }
            }
          }
        }

        /* if(isIterative) { */
        /*   for (auto &F : M) { */
        /*     if(FunctionsWithKernelLaunches.find(&F) != FunctionsWithKernelLaunches.end()){ */
        /*     LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo(); */
        /*     ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>(F).getSE(); */
        /*       computeAndPerformPlacementIterative(&F, CallInsts, LI, SE); */
        /*     } */
        /*   } */
        /* } */

      return true;
    }

    bool runOnFunctionImpl(Function &F) {

      // TODO: avoid run on stubs, but not by checking name for substring "stub"
      if (F.getName().contains("stub")) {
        errs() << "not running on " << F.getName() << "\n";
        return false;
      }

      // LLVMContext &Ctx = F.getContext();

      // Collect information about memory allocations.
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *CI = dyn_cast<CallBase>(&I)) {
            auto *Callee = CI->getCalledFunction();
            /* errs() << Callee->getName() << "\n"; */
            if (Callee && Callee->getName() == "cudaMallocManaged") {
              processMemoryAllocation(CI);
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

      // Insert memory advise APIs

      errs() << "\nKERNEL ARG TO STORES MAP\n";
      for (auto I = KernelArgToStoreMap.begin(); I != KernelArgToStoreMap.end();
          I++) {
        I->first->dump();
        for (auto U = (I->second).begin(); U != (I->second).end(); U++) {
          (*U)->dump();
        }
      }

      errs() << "\nKERNEL INVOCATION TO KERNEL ARG STRUCT MAP\n";
      for (auto I = KernelInvocationToStructMap.begin();
          I != KernelInvocationToStructMap.end(); I++) {
        errs() << "\n";
        I->first->dump();
        I->second->dump();
      }

      return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
    }
  };

} // namespace

char CudaHostTransform::ID = 0;
static RegisterPass<CudaHostTransform> X("CudaHostTransform",
    "CudaHostTransform Pass", true, true);
