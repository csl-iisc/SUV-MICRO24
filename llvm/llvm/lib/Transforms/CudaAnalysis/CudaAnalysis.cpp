//===- CudaAnalysis.cpp - Extracting access information from CUDA kernels
//---------------===//
//
// We borrow heavily from SC'19 paper
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <bits/types/FILE.h>
#include <cstddef>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <shared_mutex>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include <sstream>


using namespace llvm;

#define DEBUG_TYPE "CudaAnalysis"

static unsigned AccessID= 0;

namespace {
using std::vector;

enum AxisValueType {
  AXIS_TYPE_LOOPVAR,
  AXIS_TYPE_BIDX,
  AXIS_TYPE_BIDY,
  AXIS_TYPE_BDIMX,
  AXIS_TYPE_BDIMY,
  AXIS_TYPE_TIDX,
  AXIS_TYPE_TIDY,
  AXIS_TYPE_NONE,
};

char const *AxisValueNames[]{"LOOPVAR", "BIDX", "BIDY", "BDIMX", "BDIMY", "TIDX", "TIDY", "NONE"};

enum SpecialValueType {
  SREG_TIDX,
  SREG_TIDY,
  SREG_TIDZ,
  SREG_BIDX,
  SREG_BIDY,
  SREG_BIDZ,
  SREG_BDIMX,
  SREG_BDIMY,
  SREG_BDIMZ,
  SREG_GDIMX,
  SREG_GDIMY,
  SREG_GDIMZ,
  SREG_COMPOUND,
  SREG_NONE,
};

char const *SpecialValueNames[] = {
    "SREG_TIDX",  "SREG_TIDY",  "SREG_TIDZ",     "SREG_BIDX",  "SREG_BIDY",
    "SREG_BIDZ",  "SREG_BDIMX", "SREG_BDIMY",    "SREG_BDIMZ", "SREG_GDIMX",
    "SREG_GDIMY", "SREG_GDIMZ", "SREG_COMPOUND", "SREG_NONE",
};

struct CudaAnalysisSupportStruct {
  unsigned long int Loads;
};

enum ArgType {
  ArgTypePrimitive,
  ArgTypePointer,
  ArgTypeStructByValue,
  ArgTypeMAX,
};

char const *ArgTypeNames[] = {
  "ArgTypePrimitive",
  "ArgTypePointer",
  "ArgTypeStructByValue",
  "ArgTypeMAX",
};

struct TermAnalysisSupportStruct {
  std::set<Value *> AddOps;
  std::set<Value *> Terms;
  TermAnalysisSupportStruct(std::set<Value *> AddOps, std::set<Value *> Terms) {
    this->AddOps = AddOps;
    this->Terms = Terms;
  }
};

bool isSpecialRegisterRead(CallInst *CI) {
  auto *Callee = CI->getCalledFunction();
  if (Callee && Callee->getName().startswith("llvm.nvvm.read.ptx.sreg")) {
    return true;
  }
  return false;
}

// CudaAnalysis - The first implementation, without getAnalysisUsage.
struct CudaAnalysis : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
                  //
  std::vector<Instruction *> SeenPhiNodes;
  std::vector<Value *> SharedMemoryPointers;
  std::vector<Value *> KernelArgVector;
  std::set<Value *> TerminalValues; // terminal values are like blockIdx, kernel-arguments, 

  std::map<Value*, ArgType> KernelArgTypeMap;
  std::map<Value*, CudaAnalysisSupportStruct *> PointerInfoMap;
  std::set<Value*> ByValArgsSet;
  std::set<Value*> StackAccesses;

  std::map<Loop *, unsigned long int> LoopToIterMapping;
  std::map<Loop *, unsigned long int> LoopToTotalIterMapping;
  std::map<Loop *, unsigned long int> LoopToLoopIdMapping;
  std::map<Loop *, Loop *> LoopToParentMapping;
  std::map<Loop *, std::vector<Value*>> LoopToInitialMap;
  std::map<Loop *, std::vector<Value*>> LoopToFinalMap;
  std::map<Loop *, std::vector<Value*>> LoopToStepMap;
  std::map<Loop *, bool> LoopToInitialComputabilityMap; // poor naming: returns tracks the opposite 
  std::map<Loop *, bool> LoopToFinalComputabilityMap;
  std::map<Loop *, bool> LoopToStepComputabilityMap;
  std::map<Loop *, unsigned int> LoopToInitialValue;
  std::map<Loop *, unsigned int> LoopToFinalValue;
  std::map<Loop *, unsigned int> LoopToStepValue;

  std::map<Value *, SpecialValueType> SpecialValues;
  std::map<Value *, SpecialValueType> GridDimValues;
  std::map<Value *, AxisValueType> AxisValues;
  std::map<Value *, TermAnalysisSupportStruct *> MemoryOpToStrideInfoMap;
  std::set<Value *> LoopInductionVariables;
  std::map<Value *, Loop *> LoopInductionVariableToLoopMap;
  std::map<Value *, std::vector<Value *>> IndexSubComputationToMultiplierMap;
  std::map<Value *, std::map<Value *, std::vector<Value *>>>
      MemoryOpToMuliplierVectorMap;
  std::map<Value *, unsigned> MemoryOpToNumAccessMap;
  std::map<Value *, unsigned> MemoryOpToAccessIDMap;
  std::map<Value *, Loop*> MemoryOpToEnclosingLoopMap;
  std::map<Value *, Value*> MemoryOpToPointerMap;

  std::map<Value *, unsigned long int> BranchToBranchIdMapping;
  std::map<Value *, bool> BranchProcessed;
  std::map<Value *, Value*> MemoryOpToIfBranch;
  std::map<Value *, bool> MemoryOpToIfType;

  std::map<PHINode*, unsigned> PhiNodeToUIDMap;
  unsigned int phiNodeUIDCounter = 0;

  std::ofstream AccessDetailFile; // ("access_detail_file.lst");
  std::ofstream AccessTreeFile; // ("access_tree_file.lst");
  std::ofstream ReuseDetailfile;  // ("reuse_detail_file.lst");
  std::ofstream LoopDetailFile;  // ("loop_detail_file.lst");
  std::ofstream IfDetailFile;  // ("if_detail_file.lst");
  std::ofstream SelectOpsFile;  // ("select_ops_file.lst");
  std::ofstream SmallTressFile;  // ("small_trees_file.lst");
  std::ofstream PhiToLoopFile;  // ("phi_loop_file.lst");

  unsigned long int LoopId = 0;
  unsigned long int BranchId = 0;

  CudaAnalysis() : FunctionPass(ID) {}

  void printBack(GetElementPtrInst *G);
  void recursivePrintBack(Instruction *I);
  // bool isSpecialRegisterRead(CallInst *V);
  bool isPhiNode(Instruction *I);
  bool isSharedMemoryAccess(Value *);

  std::string getMultiplierString(Value *Multiplier);
  GetElementPtrInst *recursiveFindGep(Value *V);
  Value *recursiveFindPointer(Value *V);

  void collectTerminalSources(Value *V);
  bool isDependent(Value *V, Value *U);
  bool isTerminalValue(Value *V);

  std::string convertValueToString(Value * V);
  vector<std::string> convertValuesToStrings(std::vector<Value *> Values);
  void serializeExpressionTree(Value* V, std::ostringstream &out, std::set<Value*> PhiNodesVisited);

  std::vector<Value*> getExpressionTree(Value *V);
  bool isPointerChase(Value* V); // do not use this. this is wrong.
  bool isPointerChaseFixed(Value* V);

  Value *handleNonConstantLoopBound(Value *V);
  std::vector<Value*> handleNonConstantLoopBoundDFS(Value *V);
  Value* findPointerForGivenOp(Value* V);
  bool isDataDependent(Value *V);
  unsigned long handleNonConstantLoopBounds(Loop::LoopBounds &LI);
  bool computeIterations(LoopInfo &LI, ScalarEvolution &SE, Function &F);
  bool computeIterationsUsingBFI(LoopInfo &LI, ScalarEvolution &SE,
                                 Function &F);
  // bool findNumberOfAccesses(LoopInfo &LI, Function &F);
  bool findNumberOfAccesses(LoopInfo &LI, Function &F, std::vector<Value *>);
  bool countMemoryOperations(LoopInfo &LI, Function &F, std::vector<Value *>);
  bool findNumberOfAccessesUsingBFI(LoopInfo &LI, Function &F,
                                    std::vector<Value *>, ScalarEvolution &SE);
  bool findDirectionOfAccess(LoopInfo &LI, Function &F, std::vector<Value *>);
  bool computeStrides();
  void printTreeForStrideComputation(Value *V);
  bool findSpecialValues(Function &F);

  void computeReuse(LoopInfo &LI, Function &F, std::vector<Value *>);

  bool isFormedFromTerminalsOnly(Value *value);

  void analyzeGEPIndex(Value *MemInst, Value *Index);
  void analyzeGEPIndexForReuse(Value *MemInst, GetElementPtrInst *GEPI,
                               Value *Index);

  Value* getIndirectMemop(Value* V);

  bool isInsideConditional(Value *V);
  bool isIfConditionalBB(BasicBlock* BB);

  bool doInitialization(Module &M) override {
    AccessDetailFile.open("access_detail_file.lst");
    AccessTreeFile.open("access_tree_file.lst");
    ReuseDetailfile.open("reuse_detail_file.lst");
    LoopDetailFile.open("loop_detail_file.lst");
    IfDetailFile.open("if_detail_file.lst");
    SelectOpsFile.open("select_ops_file.lst");
    SmallTressFile.open("small_trees_file.lst");
    PhiToLoopFile.open("phi_loop_file.lst");
    return false;
  }

  bool runOnFunction(Function &F) override {
    PointerInfoMap.clear();
    KernelArgVector.clear();
    MemoryOpToNumAccessMap.clear();
    MemoryOpToAccessIDMap.clear();
    MemoryOpToEnclosingLoopMap.clear();
    MemoryOpToPointerMap.clear();
    MemoryOpToMuliplierVectorMap.clear();
    LoopToIterMapping.clear();
    LoopToInitialMap.clear();
    LoopToFinalMap.clear();
    LoopToStepMap.clear();
    LoopToInitialValue.clear();
    LoopToFinalValue.clear();
    LoopToStepValue.clear();
    PhiNodeToUIDMap.clear();

    errs() << "Kernel CudaAnalysis: ";
    errs().write_escaped(F.getName()) << '\n';
    // errs() << "Kernel arguments : \n";

    if(F.getName().compare("_Z7getNodejbP12_PixelOfNode")==0) return false;
    if(F.getName().compare("_Z11getChildrenjbP16_PixelOfChildren")==0) return false;
    if(F.getName().compare("_Z10set_resultjP11_MatchCoordiiii")==0) return false;
    if(F.getName().compare("_Z17mummergpuRCKernelP10MatchCoordPcPKiS3_ii")==0) return false;
    if(F.getName().compare("_Z11printKernelP9MatchInfoiP9AlignmentPcP12_PixelOfNodeP16_PixelOfChildrenPKiS9_iiiii")==0) return false;
    if(F.getName().compare("_Z7addr2idj")==0) return false;
    if(F.getName().compare("_Z7id2addri")==0) return false;
    if(F.getName().compare("_Z14arrayToAddress6uchar4Rj")==0) return false;
    if(F.getName().compare("_Z7getRef_iPc")==0) return false;
    if(F.getName().compare("_Z2rcc")==0) return false;

    for (auto &Arg : F.args()) {
      Arg.dump();
      Arg.getType()->dump();
      KernelArgVector.push_back(&Arg);
      if (isa<llvm::PointerType>(Arg.getType())) {
        if (Arg.hasByValAttr()) {
          // Arg.getParamByValType()->dump();
          if (Arg.getParamByValType()->isStructTy()) {
             errs() << "struct type\n";
             Arg.getParamByValType()->dump();
             ByValArgsSet.insert(&Arg);
          }
          KernelArgTypeMap[&Arg] = ArgTypeStructByValue;
        } else {
          // errs() << "pointer type\n";
          Value *PtrArg = dyn_cast<Value>(&Arg);
          PointerInfoMap[PtrArg] = new CudaAnalysisSupportStruct;
          PointerInfoMap[PtrArg]->Loads = 0;
          KernelArgTypeMap[&Arg] = ArgTypePointer;
        }
      } else {
        // errs() << "primitive type\n";
          KernelArgTypeMap[&Arg] = ArgTypePrimitive;
      }
      TerminalValues.insert(&Arg);
    }

    // errs() << "ARGUMENTS of " << &F << "\n";

    for (auto ArgIter = KernelArgVector.begin();
         ArgIter != KernelArgVector.end(); ArgIter++) {
      // (*ArgIter)->dump();
      // errs() << ArgTypeNames[KernelArgTypeMap[*ArgIter]] << "\n";
      
    }

    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();

    findSpecialValues(F);
    errs() << "TERMINAL VALUES\n";
    for (auto TermIter = TerminalValues.begin();
         TermIter != TerminalValues.end(); TermIter++) {
      (*TermIter)->dump();
      // errs() << ArgTypeNames[KernelArgTypeMap[*ArgIter]] << "\n";
    }
      
    computeIterations(LI, SE, F);
    countMemoryOperations(LI, F, KernelArgVector);

    /* findDirectionOfAccess(LI, F, KernelArgVector); */
    /* computeReuse(LI, F, KernelArgVector); */

    // computeIterationsUsingBFI(LI, SE, F);
    // findNumberOfAccessesUsingBFI(LI, F, KernelArgVector, SE);
    // computeStrides();
    // findNumberOfAccesses(LI, F, KernelArgVector);
    //
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
  }
};

bool CudaAnalysis::isIfConditionalBB(BasicBlock* BB) {
  Instruction* term = BB->getTerminator();
  if(isa<BranchInst>(term)){
    return true;
  }
  else {
    return false;
  }
}

//if the predecessor of the basic block ends with a conditional branch
bool CudaAnalysis::isInsideConditional(Value* V) {
  // TODO:  assert V is load or store
  auto I = dyn_cast<Instruction> (V);
  auto B = I->getParent();
  auto pred = B->getSinglePredecessor();
  B->dump();
  if(pred) {
    if(isIfConditionalBB(pred)) {
      errs() << "if ending bb\n";
      /* pred->dump(); */
      Instruction* cond = pred->getTerminator();
      /* cond->getOperand(0)->dump(); */
      /* cond->getOperand(1)->dump(); */
      /* cond->getOperand(2)->dump(); */
      // Extermely curiously, the order in IR and the order of operands not match up!
      if(cond->getOperand(1) == B) {
        errs() << "false\n";
        cond->dump();
        MemoryOpToIfBranch[I] = cond;
        MemoryOpToIfType[I] = false;
      } 
      if(cond->getOperand(2) == B) {
        errs() << "true\n";
        cond->dump();
        MemoryOpToIfBranch[I] = cond;
        MemoryOpToIfType[I] = true;
      } 
      if(BranchToBranchIdMapping.find(cond) == BranchToBranchIdMapping.end()){
        errs() << "new branch ID\n";
        BranchToBranchIdMapping[cond] = ++ BranchId;
        cond->getOperand(0)->dump();
        BranchProcessed[cond] = false;
        // recuse with select operations
      }
    }
  }
  /* for (auto it = pred_begin(B), et = pred_end(B); it != et; ++it) */
  /* { */
  /*   BasicBlock* predecessor = *it; */
  /*   if(isIfConditionalBB(predecessor)) { */
  /*     errs() << "if ending bb\n"; */
  /*   } */
  /* } */
  return false;
}

void CudaAnalysis::recursivePrintBack(Instruction *I) {
  I->dump();
  for (auto *OpIter = I->op_begin(); OpIter != I->op_end(); OpIter++) {
    Value *V = dyn_cast<Value>(*OpIter);
    /* V->dump(); */
    Instruction *In = dyn_cast<Instruction>(*OpIter);
    if (In) {
      if (auto *CI = dyn_cast<CallInst>(In)) {
        if (isSpecialRegisterRead(CI)) {
          errs() << "register read\n";
          return;
        }
      }
      if (isPhiNode(In)) {
        errs() << "PHI node\n";
        /* In->dump(); */
        auto It = std::find(SeenPhiNodes.begin(), SeenPhiNodes.end(), In);
        if (It != SeenPhiNodes.end()) {
          errs() << "found in list\n";
          return;
        }
        errs() << "NOT found in list\n";
        SeenPhiNodes.push_back(In);
        recursivePrintBack(In);
      }
      recursivePrintBack(In);
    }
  }
}

void CudaAnalysis::printBack(GetElementPtrInst *G) {
  errs() << "examination\n";
  errs() << "pointer : ";
  G->getPointerOperand()->dump();
  for (auto *OpIter = G->idx_begin(); OpIter != G->idx_end(); OpIter++) {
    Value *V = dyn_cast<Value>(*OpIter);
    Instruction *I = dyn_cast<Instruction>(*OpIter);
    if (I) {
      errs() << "found instruction, initiating recursive print back\n";
      SeenPhiNodes.clear();
      recursivePrintBack(I);
    }
  }
}

// we are assuming one of the operands to be const or blockid or the entire tree be made of terminal leaves
std::string CudaAnalysis::getMultiplierString(Value *Multiplier) {
  Multiplier->dump();
  assert(isa<BinaryOperator>(Multiplier));
  if (auto *MulInst = dyn_cast<BinaryOperator>(Multiplier)) {
    Value *Oper0 = MulInst->getOperand(0);
    Value *Oper1 = MulInst->getOperand(1);
    Value *ConstOper = nullptr;
    // change to only DIM values
    if ((GridDimValues.find(Oper0) != GridDimValues.end()) ||
        isa<ConstantInt>(Oper0)) {
      ConstOper = Oper0;
    }
    if ((GridDimValues.find(Oper1) != GridDimValues.end()) ||
        isa<ConstantInt>(Oper1)) {
      assert(ConstOper == nullptr);
      ConstOper = Oper1;
    }
    // ConstOper->dump();
    if(ConstOper != nullptr) {
      if (MulInst->getOpcode() == Instruction::Mul) {
        if(auto *ConstInt = dyn_cast<ConstantInt>(ConstOper)){
          return Twine(ConstInt->getSExtValue()).str();
        }
        if(GridDimValues.find(ConstOper) != GridDimValues.end()){
          return SpecialValueNames[SpecialValues[ConstOper]];
        }
      }
      if (MulInst->getOpcode() == Instruction::Shl) {
        if(auto *ConstInt = dyn_cast<ConstantInt>(ConstOper)){
          return Twine(1 << (ConstInt->getSExtValue())).str();
        }
        // TODO: MAJOR BUG: must be 2^ thing. For now we assume this path is never taken.
        if(GridDimValues.find(ConstOper) != GridDimValues.end()){
          static_assert(1, "This part is not implemented yet. Minor fix\n");
          return SpecialValueNames[SpecialValues[ConstOper]];
        }
      }
    }
    else { // need to do DFS 
      auto RPN = handleNonConstantLoopBoundDFS(Multiplier);
      auto multiplierStringVector = convertValuesToStrings(RPN);
      std::string multiplierString("");
      for (auto ArgIter = multiplierStringVector.begin(); ArgIter != multiplierStringVector.end(); ArgIter++){
        multiplierString.append(*ArgIter);
        multiplierString.append(" ");
      }
      return multiplierString;
    }
  }
  return std::string("ERROR");
}

bool CudaAnalysis::isSharedMemoryAccess(Value *V) {
  if (isa<llvm::PointerType>(V->getType())) {
    // errs() << "dumping\n";
    V->dump();
    // V->getType()->dump();
    if (auto *U = dyn_cast<User>(V)) {
      // errs() << "user \n";
      // U->dump();
      // U->getOperand(0)->dump();
      // if (isa<AddrSpaceCastInst>(U->getOperand(0))) {
      if (isa<AddrSpaceCastInst>(V)) {
        // errs() << "address space cast inst\n";
      } else {
        // errs() << "not address space cast inst\n";
      }
      if(isa<AllocaInst>(V)) {
        return false;
      }
      PointerType *Ptrtype = dyn_cast<PointerType>(U->getOperand(0)->getType());
      if (Ptrtype->isOpaque()) {
        // errs() << "opaque\n";
      }
      // errs() << "address space is : " << Ptrtype->getAddressSpace() << "\n";
      return Ptrtype->getAddressSpace() == 3;
    }
    // PointerType *Ptrtype = dyn_cast<PointerType>(V->getType());
    // if (Ptrtype->isOpaque()) {
    //   errs() << "opaque\n";
    // }
    // errs() << "address space is : " << Ptrtype->getAddressSpace() << "\n";
    // return Ptrtype->getAddressSpace() == 3;
    }
    return false;
  }

  Value *CudaAnalysis::recursiveFindPointer(Value *V) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      if(isa<GetElementPtrInst>(GEP->getPointerOperand())){
        return recursiveFindPointer(GEP->getPointerOperand());
      }
      else{
        return GEP->getPointerOperand();
      }
    }
    if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(V)) {
      return recursiveFindPointer(ASCI->getPointerOperand());
    }
    if (auto *PhiInst = dyn_cast<PHINode>(V)) {
      return findPointerForGivenOp(V); // stack based approach
    }
    return V;
  }

GetElementPtrInst *CudaAnalysis::recursiveFindGep(Value *V) {
  if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
    return GEP;
  }
  if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(V)) {
    return recursiveFindGep(ASCI->getPointerOperand());
  }
  return nullptr;
}

bool CudaAnalysis::isPhiNode(Instruction *I) {
  PHINode *Phi = dyn_cast<PHINode>(I);
  if (Phi) {
    return true;
  }
  return false;
}

bool CudaAnalysis::computeIterationsUsingBFI(LoopInfo &LI, ScalarEvolution &SE,
                                             Function &F) {
  auto BPI = BranchProbabilityInfo(F, LI);
  auto BFI = BlockFrequencyInfo(F, BPI, LI);
  for (auto &BB : F) {
    BB.dump();
    errs() << BFI.getBlockFreq(&BB).getFrequency();
    errs() << "\n\n";
  }
  return false;
}

bool CudaAnalysis::isTerminalValue(Value *V) {
  // check whether a value is an argument
  auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), V);
  if (It != KernelArgVector.end()) {
    return true;
  }
  // check whether value is constant
  if (isa<ConstantInt>(V)) {
    return true;
  }
  return false;
}

// NOTE: must have identified all terminals before calling this function
void CudaAnalysis::collectTerminalSources(Value *V) {
  errs() << "Identifying terminal sources for \n";
  V->dump();
  std::stack<Value *> ValueQueue;
  std::set<Value *> Visited;
  ValueQueue.push(V);
  Value *Top;
  while (!ValueQueue.empty()) {
    Top = ValueQueue.top();
    // Top->dump();
    ValueQueue.pop();
    Visited.insert(Top);
    if (auto *In = dyn_cast<Instruction>(Top)) {
      for (auto &Operand : In->operands()) {
        // Check if terminal
        Value *U = dyn_cast<Value>(&Operand);
        if (isTerminalValue(U)) {
          errs() << "Found Terminal \n";
          Operand->dump();
        } else {
          ValueQueue.push(U);
        }
      }
    }
  }
  return;
}

// Check if V is dependent on U
bool CudaAnalysis::isDependent(Value *V, Value *U) {
  errs() << "Checking dependency\n";
  V->dump();
  U->dump();
  std::stack<Value *> ValueQueue;
  std::set<Value *> Visited;
  ValueQueue.push(V);
  Value *Top;
  while (!ValueQueue.empty()) {
    Top = ValueQueue.top();
    Top->dump();
    ValueQueue.pop();
    if (Visited.find(Top) != Visited.end()) {
      continue;
    }
    Visited.insert(Top);
    if (auto *In = dyn_cast<Instruction>(Top)) {
      In->dump();
      for (auto &Operand : In->operands()) {
        Value *O = dyn_cast<Value>(&Operand);
        O->dump();
        if (O == U) {
          errs() << "Found U \n";
          Operand->dump();
          return true;
        }
        ValueQueue.push(O);
      }
    }
  }
  return false;
}

unsigned long CudaAnalysis::handleNonConstantLoopBounds(Loop::LoopBounds &lpb) {
  errs() << "hello from non constant loop bound handler\n";
  Value &VInitial = lpb.getInitialIVValue();
  Value &VFinal = lpb.getFinalIVValue();
  Value *VSteps = lpb.getStepValue();

  VInitial.dump();
  VFinal.dump();
  VSteps->dump();

  auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), &VInitial);
  if (It != KernelArgVector.end()) {
    errs() << "FOUND KERNEL ARG for non constant loop bound\n";
  }

  // check for VFinal being dependent on VInitial by iterating over non-terminal
  // operands recursively.
  if (isDependent(&VFinal, &VInitial)) {
    if (auto *In = dyn_cast<Instruction>(&VFinal)) {
      if (In->getOpcode() == Instruction::Add) {
        for (auto &Operand : In->operands()) {
          if (dyn_cast<Value>(&Operand) == &VInitial) {
            errs() << "ADD\n";
            In->dump();
          }
        }
      } else if (In->getOpcode() == Instruction::Mul) {
        errs() << "MUL\n";
        In->dump();
      } else {
        errs() << "UNHANDLED PATTERN\n";
      }
    }
    // collectTerminalSources(&VFinal);
  }

  // TODO: check for VInitial and VFinal being multiples or strides of each other.

  return 0;
}

bool CudaAnalysis::isPointerChaseFixed(Value* V) {
  std::stack<Value*> Stack;
  std::set<Value*> Visited;
  std::set<Value*> PhiNodesVisited;

  errs() << "is pointer chase fixing\n";
  Stack.push(V);

  bool foundOnce = false;

  while (!Stack.empty()) {
    Value *Current = Stack.top();
    Current->dump();
    Stack.pop();
    if(Current == V) {
      if (foundOnce == true) {
        errs() << "is indeed PC: ";
        V->dump();
        return true;
      } else {
        foundOnce = true;
      }
    }
    if(PhiNodesVisited.find(Current) != PhiNodesVisited.end()) {
      continue;
    }
    if(Visited.find(Current) != Visited.end()) {
      errs() << "hi visited already\n";
    /*   continue; */
    }
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
        // if the pointer operand is one of the arguments:
        // then skip it, 
        // else if the gep leads to another 
        if(PointerInfoMap.find(GEPI->getPointerOperand()) != PointerInfoMap.end()){
          errs() << "found in pim " ;
          GEPI->getPointerOperand()->dump();
          for(int i = 1; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
            Stack.push(GEPI->getOperand(i));
          }
        } else {
          errs() << "not found in pim " ;
          for(int i = 0; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
            Stack.push(GEPI->getOperand(i));
          }
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
  return false;
}

// recursively traverses the expression tree and checks for pointer chase loops
bool CudaAnalysis::isPointerChase(Value* V) {
  errs() << "\nstart is pointer chase : ";
  V->dump();

  std::stack<Value*> Stack;
  std::set<Value*> Visited;
  std::set<Value*> OnStack;
  std::set<Value*> PhiNodesVisited;

  Stack.push(V);

  while (!Stack.empty()) {
    Value *Current = Stack.top();
    errs() << "top: ";
    Current->dump();
    if(Visited.find(Current) == Visited.end()){
      OnStack.insert(Current);
      Visited.insert(Current);
    }
    else {
      OnStack.erase(Current);
      Stack.pop();
    }
    if(PhiNodesVisited.find(Current) != PhiNodesVisited.end()) {
      continue;
    }
    if(TerminalValues.find(Current)!= TerminalValues.end()){
      continue;
    }
    // iterate through operands
    if (isa<Instruction>(Current)) {
      auto *In = dyn_cast<Instruction>(Current);
      if(auto * LI = dyn_cast<LoadInst>(In)) {
        /* Stack.push(LI->getPointerOperand()); */
        auto oper = LI->getPointerOperand();
          if ((OnStack.find(oper) != OnStack.end()) && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
      } else if (auto * SI = dyn_cast<StoreInst>(In)) {
        /* Stack.push(SI->getPointerOperand()); */
        auto oper = SI->getPointerOperand();
          if ((OnStack.find(oper) != OnStack.end()) && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
      } else if (auto * GEPI = dyn_cast<GetElementPtrInst>(In)) {
        // if the pointer operand is one of the arguments:
        // then skip it, 
        // else if the gep leads to another 
        if(PointerInfoMap.find(GEPI->getPointerOperand()) != PointerInfoMap.end()){
          for(int i = 1; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
            /* Stack.push(GEPI->getOperand(i)); */
            auto oper = GEPI->getOperand(i);
          if ((OnStack.find(oper) != OnStack.end()) && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
          }
        } else {
          for(int i = 0; i < GEPI->getNumIndices() + 1; i++){ // indices includes the pointer 
            /* Stack.push(GEPI->getOperand(i)); */
            auto oper = GEPI->getOperand(i);
          if ((OnStack.find(oper) != OnStack.end()) && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
          }
        }
      } else if (auto * Phi = dyn_cast<PHINode>(In)) {
        for (auto &Operand : In->operands()) {
          /* Stack.push(Operand); */
          auto oper = dyn_cast<Value>(Operand);
          if ((OnStack.find(oper) != OnStack.end()) && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
        }
        PhiNodesVisited.insert(Phi);
      } else if (auto * CI = dyn_cast<CallInst>(In)) {
        auto *Callee = CI->getCalledFunction();
        if (Callee->getName() == "llvm.nvvm.shfl.sync.idx.i32") {
          errs() << "found a special funcion\n";
          /* for(auto arg = CI->arg_begin(); arg != CI->arg_end(); arg++) { */
          /*   if(auto argval = dyn_cast<Value>(arg)){ */
          /*     argval->dump(); */
          /*   } */
          /* } */
          auto arg1 = CI->getArgOperand(1);
          auto arg1val = dyn_cast<Value>(arg1);
          arg1->dump();
          /* Stack.push(arg1val); */
          auto oper = arg1val;
          if (OnStack.find(oper) != OnStack.end() && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
        }
      } else{
        for (auto &Operand : In->operands()) {
          /* Stack.push(Operand); */
          auto oper = dyn_cast<Value>(Operand);
          if (OnStack.find(oper) != OnStack.end() && (PhiNodesVisited.find(oper) == PhiNodesVisited.end())){
            errs() << "FOUND POINTER CHASE\n";
            oper->dump();
            return true;
          } else if((Visited.find(oper) == Visited.end())){ 
            Stack.push(oper);
          }
        }
      }
      continue;
    }
  }
  errs() << "end is pointer chase\n\n";
  return false;
}

std::vector<Value *> CudaAnalysis::getExpressionTree(Value *V) {
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
    RPN.push_back(Current);
    if(Visited.find(Current) != Visited.end()) {
      errs() << "hi visited already\n";
    /*   continue; */
    }
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
        // if the pointer operand is one of the arguments:
        // then skip it, 
        // else if the gep leads to another 
        if(PointerInfoMap.find(GEPI->getPointerOperand()) != PointerInfoMap.end()){
          for(int i = 1; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
            Stack.push(GEPI->getOperand(i));
          }
        } else {
          for(int i = 0; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
            Stack.push(GEPI->getOperand(i));
          }
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

// do tree traversal and find any data dependencies
bool CudaAnalysis::isDataDependent(Value *V) {
  std::stack<Value*> Stack;
  std::set<Value*> Visited;
  std::set<Value*> PhiNodesVisited;

  errs() << "checking for data dependencies \n";
  Stack.push(V);

  while (!Stack.empty()) {
    Value *Current = Stack.top();
    Stack.pop();
    if(PhiNodesVisited.find(Current) != PhiNodesVisited.end()) {
      continue;
    }
    if(Visited.find(Current) != Visited.end()) {
      continue;
    }
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
        if(ByValArgsSet.find(GEPI->getPointerOperand())  != ByValArgsSet.end()){
          errs() << "found indirect access to struct element\n";
        }
        if(PointerInfoMap.find(GEPI->getPointerOperand())  != PointerInfoMap.end()){
          errs() << "found indirect access to kernel arg pointer\n";
          return true;
        }
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

  return false;
}

Value* CudaAnalysis::findPointerForGivenOp(Value *V) {
  std::stack<Value*> Stack;
  std::set<Value*> Visited;
  std::set<Value*> PhiNodesVisited;
  Stack.push(V);
  while (!Stack.empty()) {
    Value *Current = Stack.top();
    Current->dump();
    Stack.pop();
    if(PhiNodesVisited.find(Current) != PhiNodesVisited.end()) {
      continue;
    }
    if(Visited.find(Current) != Visited.end()) {
      errs() << "hi\n";
      continue;
    }
    if(TerminalValues.find(Current)!=TerminalValues.end()){
      errs() << "Found terminal\n";
      return Current;
    }
    // iterate through operands
    if (isa<Instruction>(Current)) {
      auto *In = dyn_cast<Instruction>(Current);
      if(auto * LI = dyn_cast<LoadInst>(In)) {
        Stack.push(LI->getPointerOperand());
      } else if (auto * SI = dyn_cast<StoreInst>(In)) {
        Stack.push(SI->getPointerOperand());
      } else if (auto * GEPI = dyn_cast<GetElementPtrInst>(In)) {
        /* for(int i = 0; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer */ 
          Stack.push(GEPI->getOperand(0));
        /* } */
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
  return nullptr;
}

std::vector<Value *> CudaAnalysis::handleNonConstantLoopBoundDFS(Value *V) {
  std::vector<Value*> RPN(0);
  std::stack<Value*> Stack;
  std::set<Value*> Visited;
  std::set<Value*> PhiNodesVisited;

  errs() << "Handling non-constant loop bound with DFS\n";
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

Value *CudaAnalysis::handleNonConstantLoopBound(Value *V) {
  errs() << "HANDLING non constant loop bound\n";
  // V->dump();
  auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), V);
  if (It != KernelArgVector.end()) {
    errs() << "FOUND KERNEL ARG for non constant loop bound\n";
    return V;
  }
  errs() << "Checking inside\n";
  if (auto *I = dyn_cast_or_null<Instruction>(V)) {
    auto Iter = std::find(KernelArgVector.begin(), KernelArgVector.end(),
                          I->getOperand(0));
    if (Iter != KernelArgVector.end()) {
      errs() << "FOUND KERNEL ARG for non constant loop bound INDIRECTLY\n";
      return I->getOperand(0);
    }
  }
  auto GDIt = GridDimValues.find(V);
  if (GDIt != GridDimValues.end()) {
    errs() << "FOUND GRID DIM value in loop bound\n";
    return V;
  }
  return nullptr;
}

std::string 
CudaAnalysis::convertValueToString(Value* V) {
    if(TerminalValues.find(V) != TerminalValues.end()){
        errs() << "terminal value\n";
        auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), V);
        if(It != KernelArgVector.end()) {
            std::string arg = "ARG";
            auto argid = It - KernelArgVector.begin();
            arg.append(std::to_string(argid));
            errs() << argid << "\n";
            return (arg);
        }
        auto AxIter = AxisValues.find(V);
        if(AxIter != AxisValues.end()) {
            return std::string(AxisValueNames[AxisValues[(*AxIter).first]]);
        }
    }
    if(isa<ConstantInt>(V)){
        errs() << "constant\n";
        auto Value = dyn_cast<ConstantInt>(V);
        return (std::to_string(Value->getSExtValue()));
    }
    if(isa<ConstantFP>(V)){
        errs() << "constant\n";
        auto Value = dyn_cast<ConstantFP>(V);
        std::string fpStr;
        llvm::raw_string_ostream rso(fpStr);
        Value->print(rso);
        return rso.str();
    }
    if(isa<UndefValue>(V)){
        errs() << "undef value\n";
        auto Value = dyn_cast<UndefValue>(V);
        return std::string("UNDEF");
    }
    if(isa<Instruction>(V)) {
        errs() << "instruction\n";
        auto In = dyn_cast<Instruction>(V);
        errs() << In->getOpcodeName() << "\n";
        if (isa<BinaryOperator>(In)) {
            if (In->getOpcode() == Instruction::Add) {
                errs() << "add\n";
                return std::string("ADD");
            }
            if (In->getOpcode() == Instruction::Sub) {
                errs() << "sub\n";
                return std::string("SUB");
            }
            if (In->getOpcode() == Instruction::Or) {
                errs() << "or\n";
                return std::string("OR");
            }
            if (In->getOpcode() == Instruction::And) {
                errs() << "and\n";
                return std::string("AND");
            }
            if (In->getOpcode() == Instruction::Mul) {
                errs() << "Mul\n";
                return std::string("MUL");
            }
            if (In->getOpcode() == Instruction::UDiv) {
                errs() << "UDiv\n";
                return std::string("UDIV");
            }
            if (In->getOpcode() == Instruction::SDiv) {
                errs() << "SDiv\n";
                return std::string("SDIV");
            }
            if (In->getOpcode() == Instruction::SRem) {
                errs() << "SRem\n";
                return std::string("SREM");
            }
            if (In->getOpcode() == Instruction::Shl) {
                errs() << "Shl\n";
                return std::string("SHL");
            }
            if (In->getOpcode() == Instruction::LShr) {
                errs() << "LShr\n";
                return std::string("LSHR");
            }
            if (In->getOpcode() == Instruction::Xor) {
                errs() << "Xor\n";
                return std::string("XOR");
            }
            if (In->getOpcode() == Instruction::FDiv) {
                errs() << "FDiv\n";
                return std::string("FDIV");
            }
            if (In->getOpcode() == Instruction::FMul) {
                errs() << "FMul\n";
                return std::string("FMUL");
            }
        }
        if (In->getOpcode() == Instruction::ICmp) {
            errs() << "Icmp\n";
            return std::string("ICMP");
        }
        if (In->getOpcode() == Instruction::FCmp) {
            errs() << "Fcmp\n";
            return std::string("FCMP");
        }
        if (In->getOpcode() == Instruction::FPToSI) {
            errs() << "FPToSI\n";
            return std::string("FPTOSI");
        }
        if (In->getOpcode() == Instruction::UIToFP) {
            errs() << "UIToFP\n";
            return std::string("UITOFP");
        }
        if (In->getOpcode() == Instruction::SIToFP) {
            errs() << "SIToFP\n";
            return std::string("SITOFP");
        }
        if (In->getOpcode() == Instruction::Call) {
            errs() << "Call\n";
            return std::string("CALL");
        }
        if (In->getOpcode() == Instruction::AtomicRMW) {
            errs() << "AtomicRmw\n";
            return std::string("ATOMICRMW");
        }
        if (In->getOpcode() == Instruction::PHI) {
            errs() << "Phi\n";
            unsigned phiNodeUID = 0;
            PHINode* PhiNodeInst = dyn_cast<PHINode>(In);
            if(PhiNodeToUIDMap.find(PhiNodeInst) != PhiNodeToUIDMap.end()) {
                phiNodeUID = PhiNodeToUIDMap[PhiNodeInst];
            } else {
                phiNodeUID = ++phiNodeUIDCounter;
                PhiNodeToUIDMap[PhiNodeInst] = phiNodeUID;
            }
            return std::string("PHI"+std::to_string(phiNodeUID));
        }
        if (In->getOpcode() == Instruction::Load) {
            errs() << "LOAD\n";
            return std::string("LOAD");
        }
        if (In->getOpcode() == Instruction::Store) {
            errs() << "STORE\n";
            return std::string("STORE");
        }
        if (In->getOpcode() == Instruction::GetElementPtr) {
            errs() << "GEP\n";
            return std::string("GEP");
        }
        if (In->getOpcode() == Instruction::ZExt) {
            errs() << "ZEXT\n";
            return std::string("ZEXT");
        }
        if (In->getOpcode() == Instruction::SExt) {
            errs() << "SEXT\n";
            return std::string("SEXT");
        }
        if (In->getOpcode() == Instruction::Freeze) {
            errs() << "FREEZE\n";
            return std::string("FREEZE");
        }
        if (In->getOpcode() == Instruction::Trunc) {
            errs() << "TRUNC\n";
            return std::string("TRUNC");
        }
        if (In->getOpcode() == Instruction::Select) {
            errs() << "SELECT\n";
            return std::string("SELECT");
        }
    }
    errs() << "not handled\n";
    V->dump();
    /* assert(false); */
    return std::string("UNKNOWN");
}



vector<std::string>
CudaAnalysis::convertValuesToStrings(std::vector<Value *> Values) {
  errs() << " \nconvert values to strings\n";
  std::vector<std::string> Strings;
  for (auto ValueIter = Values.begin(); ValueIter != Values.end();
      ValueIter++) {
    (*ValueIter)->dump();
    if(TerminalValues.find(*ValueIter) != TerminalValues.end()){
      errs() << "terminal value\n";
      auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), *ValueIter);
      if(It != KernelArgVector.end()) {
        std::string arg = "ARG";
        auto argid = It - KernelArgVector.begin();
        arg.append(std::to_string(argid));
        errs() << argid << "\n";
        Strings.push_back(arg);
        continue;
      }
      auto AxIter = AxisValues.find(*ValueIter);
      if(AxIter != AxisValues.end()) {
        Strings.push_back(AxisValueNames[AxisValues[(*AxIter).first]]);
      }
    }
    if(isa<ConstantInt>(*ValueIter)){
      errs() << "constant\n";
      auto Value = dyn_cast<ConstantInt>(*ValueIter);
      Strings.push_back(std::to_string(Value->getSExtValue()));
    }
    if(isa<Instruction>(*ValueIter)) {
      errs() << "instruction\n";
      auto In = dyn_cast<Instruction>(*ValueIter);
      errs() << In->getOpcodeName() << "\n";
      if (isa<BinaryOperator>(In)) {
        if (In->getOpcode() == Instruction::Add) {
          errs() << "add\n";
          Strings.push_back("ADD");
        }
        if (In->getOpcode() == Instruction::Sub) {
          errs() << "sub\n";
          Strings.push_back("SUB");
        }
        if (In->getOpcode() == Instruction::Or) {
          errs() << "or\n";
          Strings.push_back("OR");
        }
        if (In->getOpcode() == Instruction::And) {
          errs() << "and\n";
          Strings.push_back("AND");
        }
        if (In->getOpcode() == Instruction::Mul) {
          errs() << "Mul\n";
          Strings.push_back("MUL");
        }
        if (In->getOpcode() == Instruction::UDiv) {
          errs() << "UDiv\n";
          Strings.push_back("UDIV");
        }
        if (In->getOpcode() == Instruction::SDiv) {
          errs() << "SDiv\n";
          Strings.push_back("SDIV");
        }
        if (In->getOpcode() == Instruction::Shl) {
          errs() << "Shl\n";
          Strings.push_back("SHL");
        }
      }
      if (In->getOpcode() == Instruction::ICmp) {
        errs() << "Icmp\n";
        Strings.push_back("ICMP");
      }
      if (In->getOpcode() == Instruction::PHI) {
        errs() << "Phi\n";
        Strings.push_back("PHI");
      }
    }
  }
  for (auto ArgIter = Strings.begin(); ArgIter != Strings.end(); ArgIter++){
    errs() << (*ArgIter) << "  ";
  }
  return Strings;
}

void CudaAnalysis::serializeExpressionTree(Value* root, std::ostringstream &out, std::set<Value*> PhiNodesVisited) {

    if (root == nullptr)
        return;

    errs()<< "node: ";
    root->dump();

    out << " ( ";
    out << (convertValueToString(root)) << " ";

    if(PhiNodesVisited.find(root) != PhiNodesVisited.end()) {
        out << " ) ";
        return;
    }
    if(TerminalValues.find(root)!=TerminalValues.end()){
        out << " ) ";
        return;
    }

    if (isa<Instruction>(root)) {
        auto *In = dyn_cast<Instruction>(root);
        if(auto * LI = dyn_cast<LoadInst>(In)) {
            serializeExpressionTree(LI->getPointerOperand(), out, PhiNodesVisited);
        } else if (auto * SI = dyn_cast<StoreInst>(In)) {
            serializeExpressionTree(SI->getPointerOperand(), out, PhiNodesVisited);
        } else if (auto * GEPI = dyn_cast<GetElementPtrInst>(In)) {
            // if the pointer operand is one of the arguments:
            // then skip it, 
            // else if the gep leads to another 
            if(PointerInfoMap.find(GEPI->getPointerOperand()) != PointerInfoMap.end()){
                for(int i = 1; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
                    /* Stack.push(GEPI->getOperand(i)); */
                    serializeExpressionTree(GEPI->getOperand(i), out, PhiNodesVisited);
                }
            } else {
                for(int i = 0; i < GEPI->getNumIndices() + 1; i++){ // indices not includes the pointer 
                    /* Stack.push(GEPI->getOperand(i)); */
                    serializeExpressionTree(GEPI->getOperand(i), out, PhiNodesVisited);
                }
            }
        } else if (auto * Phi = dyn_cast<PHINode>(In)) {
            PhiNodesVisited.insert(Phi);
            for (auto &Operand : In->operands()) {
                /* Stack.push(Operand); */
                serializeExpressionTree(Operand, out, PhiNodesVisited);
            }
        } else{
            for (auto &Operand : In->operands()) {
                /* Stack.push(Operand); */
                serializeExpressionTree(Operand, out, PhiNodesVisited);
            }
        }
    }
    out << " ) ";
}

bool CudaAnalysis::computeIterations(LoopInfo &LI, ScalarEvolution &SE,
        Function &F) {
    errs() << "Compute Iteration\n";
    LoopToTotalIterMapping.clear(); // clear this map, since we are keeping track
                                    // on a per kernel basis.
    for (LoopInfo::iterator lii = LI.begin(); lii != LI.end(); ++lii) {
        errs() << "\nLOOP \n" << *lii << "\n";
        /* (*lii)->dump(); */
        // errs() << (*lii) << "\n";
        Loop &Root = *(*lii);
        auto *Loopnest = new LoopNest(Root, SE);
        // errs() << "num loops = " << Loopnest->getNumLoops() << "\n";
        // errs() << "depth loops = " << Loopnest->getNestDepth() << "\n";
        unsigned Loopnestdepth = Loopnest->getNestDepth();
        auto Loops = Loopnest->getLoops();
        for (const auto *Li = Loops.begin(); Li != Loops.end(); Li++) {
            (*Li)->dump();
            errs() << "is canonical " << ((*Li)->isCanonical(SE)) << "\n";
            /* errs() << "loop details " << (*Li) << "\n"; */
            /* errs() << "loop parent " << (*Li)->getParentLoop() << "\n"; */
            /* errs() << "Induciton variable is :"; */
            /* auto *LIV = (*Li)->getCanonicalInductionVariable(); */
            auto *LIV = (*Li)->getInductionVariable(SE);
            if (LIV) {
                errs() << "LIV: ";
                LIV->dump();
                LoopInductionVariables.insert(LIV);
                LoopInductionVariableToLoopMap[LIV] = (*Li);
                AxisValues[LIV] = AXIS_TYPE_LOOPVAR;
            }
            auto Loopbounds = (*Li)->getBounds(SE);
            if (Loopbounds) {
                errs() << "loop bounds found\n";
                // Loopbounds->getInitialIVValue().dump();
                // Loopbounds->getStepValue()->dump();
                // Loopbounds->getFinalIVValue().dump();
                Loop::LoopBounds &lpb = Loopbounds.value();
                Value &VInitial = lpb.getInitialIVValue();
                Value &VFinal = lpb.getFinalIVValue();
                Value *VSteps = lpb.getStepValue();
                bool IsConstInitial = false, IsConstFinal = false, IsConstSteps = false;
                unsigned Initial = 0, Final = 0, Steps = 0;
                std::vector<Value*> InitialRPN, FinalRPN, StepRPN;
                if (ConstantInt *CInitial = dyn_cast<ConstantInt>(&VInitial)) {
                    Initial = CInitial->getSExtValue();
                    errs() << "initial " << Initial << "\n";
                    IsConstInitial = true;
                    LoopToInitialValue[*Li] = Initial;
                } else {
                    errs() << "Initial NOT A CONSTANT\n";
                    VInitial.dump();
                    // handleNonConstantLoopBound(&VInitial);
                    InitialRPN = handleNonConstantLoopBoundDFS(&VInitial);
                    LoopToInitialComputabilityMap[*Li] = isDataDependent(&VInitial);
                    LoopToInitialMap[*Li] = InitialRPN;
                }
                if (ConstantInt *CFinal = dyn_cast<ConstantInt>(&VFinal)) {
                    Final = CFinal->getSExtValue();
                    errs() << "final " << Final << "\n";
                    IsConstFinal = true;
                    LoopToFinalValue[*Li] = Final;
                } else {
                    errs() << "Final NOT A CONSTANT\n";
                    VFinal.dump();
                    // handleNonConstantLoopBound(&VFinal);
                    FinalRPN = handleNonConstantLoopBoundDFS(&VFinal);
                    LoopToFinalComputabilityMap[*Li] = isDataDependent(&VFinal);
                    LoopToFinalMap[*Li] = FinalRPN;
                }
                if (ConstantInt *CSteps = dyn_cast<ConstantInt>(VSteps)) {
                    Steps = CSteps->getSExtValue();
                    errs() << "steps " << Steps << "\n";
                    IsConstSteps = true;
                    LoopToStepValue[*Li] = Steps;
                } else {
                    errs() << "Steps NOT A CONSTANT\n";
                    VSteps->dump();
                    // handleNonConstantLoopBound(VSteps);
                    StepRPN = handleNonConstantLoopBoundDFS(VSteps);
                    LoopToStepComputabilityMap[*Li] = isDataDependent(VSteps);
                    LoopToStepMap[*Li] = StepRPN;
                }
                if (IsConstInitial && IsConstFinal && IsConstSteps) {
                    unsigned long Iters = (Final - Initial) / Steps;
                    // errs() << "iters " << Iters << "\n";
                    LoopToIterMapping[(*Li)] = Iters;
                } else {
                    unsigned long Iters = handleNonConstantLoopBounds(lpb);
                    errs() << "iters " << Iters << "\n";
                    LoopToIterMapping[(*Li)] = Iters;
                }
            } else {
                errs() << "loop bound not found, requiring manual loop info computing\n";
                auto BB = (*Li)->getHeader();
                /* BB->dump(); */
                for (auto &I : (*BB)) {
                    if(isPhiNode(&I)) {
                        I.dump();
                    }
                }
            }
        }

    // errs() << "Base Iteration Map \n";
    for (auto i = LoopToIterMapping.begin(); i != LoopToIterMapping.end();
         i++) {
      // i->first->dump();
      // errs() << "iters  " << i->second << "\n";
    }

    // loop depth starts at 1; skip 1
    for (auto Ldepth = 1; Ldepth <= Loopnest->getNestDepth(); Ldepth++) {
      // errs() << "depth = " << ldepth << "\n";
      auto Loopsatdepth = Loopnest->getLoopsAtDepth(Ldepth);
      for (auto *Li = Loopsatdepth.begin(); Li != Loopsatdepth.end(); Li++) {
        // (*li)->dump();
        if (Ldepth == 1) {
          LoopToTotalIterMapping[(*Li)] = LoopToIterMapping[(*Li)];
        } else {
          LoopToTotalIterMapping[(*Li)] =
              LoopToIterMapping[(*Li)] *
              LoopToTotalIterMapping[(*Li)->getParentLoop()];
        }
        LoopToParentMapping[(*Li)] = (*Li)->getParentLoop();
        LoopToLoopIdMapping[(*Li)] = ++LoopId;
      }
    }
  }

  if (!LoopDetailFile.is_open()) {
    errs() << "Unable to open file\n";
  } else {
  }

  errs() << "Total Iteration Map \n";
  for (auto I = LoopToTotalIterMapping.begin();
      I != LoopToTotalIterMapping.end(); I++) {
    I->first->dump();
    errs() << "loop id = " << LoopToLoopIdMapping[I->first] << "\n";
    errs() << "iters  " << I->second << "\n";
    // BUG: if the actual number of iteration is indeed zero, then this hack
    // won't work
    /* if (I->second != 0) { */
    unsigned long int parent_id = 0;
    if(LoopToParentMapping.find(I->first) != LoopToParentMapping.end()) {
        errs() << "parent loop is " << LoopToLoopIdMapping[LoopToParentMapping[I->first]];
        parent_id = LoopToLoopIdMapping[LoopToParentMapping[I->first]];
    }
    if (0) {
      LoopDetailFile << F.getName().str() << " "
          << parent_id << " " 
        << LoopToLoopIdMapping[I->first];
      LoopDetailFile << " IT "
        << LoopToIterMapping[I->first] << "\n";
    } else {
      LoopDetailFile << F.getName().str() << " " << LoopToLoopIdMapping[I->first] << " ";
          LoopDetailFile << " " << parent_id<< " " ;
      I->first->dump();
      auto InitialRPN = convertValuesToStrings(LoopToInitialMap[I->first]);
      auto FinalRPN = convertValuesToStrings(LoopToFinalMap[I->first]);
      auto StepRPN = convertValuesToStrings(LoopToStepMap[I->first]);
      errs() << "\ninitial\n";
      if (LoopToInitialValue.find(I->first) == LoopToInitialValue.end()){
        LoopDetailFile << "IN ";
        if(LoopToInitialComputabilityMap[I->first] == true) {
          errs() << "loop is not computable initial\n";
        }
        for (auto ArgIter = InitialRPN.begin(); ArgIter != InitialRPN.end(); ArgIter++){
          errs() << (*ArgIter) << "  ";
          LoopDetailFile << (*ArgIter) << " ";
        }
      }
      else {
        LoopDetailFile << "IN ";
        LoopDetailFile << LoopToInitialValue[I->first] << " ";
        errs() << LoopToInitialValue[I->first] << "\n";
      }
      errs() << "\nfinal\n";
      if (LoopToFinalValue.find(I->first) == LoopToFinalValue.end()){
        LoopDetailFile << "FIN ";
        if(LoopToFinalComputabilityMap[I->first] == true) {
          errs() << "loop is not computable final\n";
          LoopDetailFile << "INCOMP ";
        } else {
          for (auto ArgIter = FinalRPN.begin(); ArgIter != FinalRPN.end(); ArgIter++){
            errs() << (*ArgIter) << "  ";
            LoopDetailFile << (*ArgIter) << " ";
          }
        }
      }
      else {
        LoopDetailFile << "FIN ";
        LoopDetailFile << LoopToFinalValue[I->first] << " ";
        errs() << LoopToFinalValue[I->first] << "\n";
      }
      errs() << "\nstep\n";
      if (LoopToStepValue.find(I->first) == LoopToStepValue.end()){
        LoopDetailFile << "STEP ";
        if(LoopToStepComputabilityMap[I->first] == true) {
          errs() << "loop is not computable step\n";
        }
        for (auto ArgIter = StepRPN.begin(); ArgIter != StepRPN.end(); ArgIter++){
          errs() << (*ArgIter) << "  ";
          LoopDetailFile << (*ArgIter) << " ";
        }
      }
      else {
        LoopDetailFile << "STEP ";
        errs() << LoopToStepValue[I->first] << "\n";
        LoopDetailFile << LoopToStepValue[I->first] << " ";
      }
    }
    LoopDetailFile << "\n";
  }

  return true;
}

bool CudaAnalysis::findNumberOfAccessesUsingBFI(
    LoopInfo &LI, Function &F, std::vector<Value *> KernelArgVector,
    ScalarEvolution &SE) {
  auto BPI = BranchProbabilityInfo(F, LI);
  auto BFI = BlockFrequencyInfo(F, BPI, LI);
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LdI = dyn_cast<LoadInst>(&I)) {
        // errs() << "\n";
        // LdI->dump();
        for (Use &U : LdI->operands()) {
          // U->dump();
          // if (auto G = dyn_cast<GetElementPtrInst>(U)) {
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "found GEP\n";
            unsigned Iters;
            if (BasicBlock *Parent = LdI->getParent()) {
              Iters = BFI.getBlockFreq(Parent).getFrequency();
            } else {
              Iters = 1;
            }
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "pointer address space = "
              //  << G->getPointerAddressSpace() << "\n";
              G->getPointerOperand()->dump();
              PointerInfoMap[G->getPointerOperand()]->Loads += Iters;
              MemoryOpToNumAccessMap[&I] = Iters;
              MemoryOpToAccessIDMap[&I] = AccessID++; //post increment
              // errs() << "Iters = " << Iters << "\n";
            } else {
              // errs() << "shared memory access\n";
            }
          } else {
            // errs() << "GEP not found\n";
          }
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        errs() << "\n";
        // SI->dump();
        for (Use &U : SI->operands()) {
          // U->dump();
          // if (auto G = dyn_cast<GetElementPtrInst>(U)) {
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "found GEP\n";
            unsigned Iters;
            if (BasicBlock *Parent = SI->getParent()) {
              Iters = BFI.getBlockFreq(Parent).getFrequency();
            } else {
              Iters = 1;
            }
            auto *Ptr = G->getPointerOperand();
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "pointer address space = "
              //        << G->getPointerAddressSpace() << "\n";
              // G->getPointerOperand()->dump();
              PointerInfoMap[Ptr]->Loads += Iters;
              MemoryOpToNumAccessMap[&I] = Iters;
              MemoryOpToAccessIDMap[&I] = AccessID++; //post increment
              // errs() << "Iters = " << Iters << "\n";
            } else {
              // errs() << "shared memory access\n";
            }
          } else {
            // errs() << "GEP not found\n";
          }
        }
      }
    }
  }

  // errs() << "ARGUMENTS of " << &F << "\n";
  // for (auto ArgIter = KernelArgVector.begin(); ArgIter != KernelArgVector.end();
  //      ArgIter++) {
  //   (*ArgIter)->dump();
  // }

  // errs() << "\nCounts of #access to kernel parameter arrays\n";
  // for (auto I = PointerInfoMap.begin(); I != PointerInfoMap.end(); I++) {
  //   I->first->dump();
  //   auto It =
  //       std::find(KernelArgVector.begin(), KernelArgVector.end(), I->first);
  //   errs() << "Location in kernel arg list  " << It - KernelArgVector.begin()
  //          << "\n";
  //   errs() << "Count  " << I->second->Loads << "\n";
  // }

  errs() << "Attempting file write\n";
  if (!AccessDetailFile.is_open()) {
    errs() << "Unable to open file\n";
  } else {
    for (auto I = PointerInfoMap.begin(); I != PointerInfoMap.end(); I++) {
      I->first->dump();
      auto It =
          std::find(KernelArgVector.begin(), KernelArgVector.end(), I->first);
      AccessDetailFile << F.getName().str() << " "
                       << It - KernelArgVector.begin() << " "
                       << I->second->Loads << "\n";
    }
  }
  return false;
}

void CudaAnalysis::analyzeGEPIndexForReuse(Value *MemInst,
    GetElementPtrInst *GEPI,
    Value *Index) {
  errs() << "Reuse Analysis\n";
  GEPI->dump();
  // Index->dump();
  std::stack<Value *> ValueQueue;
  std::set<Value *> Visited;
  std::vector<Value *> TopologicalOrderList;
  ValueQueue.push(Index);
  Value *Top;
  while (!ValueQueue.empty()) {
    Top = ValueQueue.top();
    errs() << "top: "; Top->dump();
    ValueQueue.pop();
    Visited.insert(Top);
    if (std::find(TopologicalOrderList.begin(), TopologicalOrderList.end(),
          Top) == TopologicalOrderList.end()) {
      TopologicalOrderList.push_back(Top);
    }
    if (auto *In = dyn_cast<Instruction>(Top)) {
      /* In->dump(); */
      if (SpecialValues.find(In) != SpecialValues.end()) {
        errs() << "special value "
          << SpecialValueNames[SpecialValues.find(In)->second] << " \n";
        In->dump();
      } else if (LoopInductionVariables.find(In) !=
          LoopInductionVariables.end()) {
        errs() << "loop induction value \n";
        In->dump();
        for (auto &Operand : In->operands()) {
          // Operand->dump();
          if (Visited.find(Operand) == Visited.end()) {
            if (isa<Instruction>(Operand)) {
              ValueQueue.push(Operand);
            }
            if( std::find(KernelArgVector.begin(), KernelArgVector.end(), Operand) != KernelArgVector.end()) {
              ValueQueue.push(Operand);
            }
          }
        }
      } else {
        for (auto &Operand : In->operands()) {
          Operand->dump();
          if (Visited.find(Operand) == Visited.end()) {
            if (isa<Instruction>(Operand)) {
              ValueQueue.push(Operand);
            }
            if( std::find(KernelArgVector.begin(), KernelArgVector.end(), Operand) != KernelArgVector.end()) {
              ValueQueue.push(Operand);
            }
          }
        }
      }
    }
  }

  std::reverse(TopologicalOrderList.begin(), TopologicalOrderList.end());
  errs() << "TOPOLOGICAL ORDER\n";
  for (auto TopoIter = TopologicalOrderList.begin();
       TopoIter != TopologicalOrderList.end(); TopoIter++) {
    (*TopoIter)->dump();
  }
  errs() << "END TOPOLOGICAL ORDER\n";

  for (auto TopoIter = TopologicalOrderList.begin();
      TopoIter != TopologicalOrderList.end(); TopoIter++) {
    /* (*TopoIter)->dump(); */
    if (AxisValues.find(*TopoIter) == AxisValues.end() &&
        std::find(KernelArgVector.begin(), KernelArgVector.end(), *TopoIter) == KernelArgVector.end()) {
      errs() << "NOT AN AXIS VALUE\n";
      continue;
    }
    errs() << "\nTOPO ITEM which is AXIS VALUE\n";
    (*TopoIter)->dump();
    std::queue<Value *> Descendents;
    std::set<Value *> LocallyVisited;
    LocallyVisited.clear();
    Descendents.push(*TopoIter);
    Value *Descendent;
    while (!Descendents.empty()) {
      errs() << "Descendent\n";
      Descendent = Descendents.front();
      Descendents.pop();
      LocallyVisited.insert(Descendent);
      Descendent->dump();
      /* if (auto *In = dyn_cast<Instruction>(Descendent)) { */
        // In->dump();
        /* for (auto *User : In->users()) { */
        for (auto *User : Descendent->users()) {
          errs() << "USERs\n";
           User->dump();
          if (IndexSubComputationToMultiplierMap.find(User) !=
              IndexSubComputationToMultiplierMap.end()) {
            if (std::find(TopologicalOrderList.begin(),
                  TopologicalOrderList.end(),
                  User) != TopologicalOrderList.end()) {
              errs() << "-----MULTIPLIER ANCESTOR\n";
              User->dump();
              MemoryOpToMuliplierVectorMap[MemInst][*TopoIter].push_back(User);
            }
          }
          if (std::find(TopologicalOrderList.begin(),
                TopologicalOrderList.end(),
                User) != TopologicalOrderList.end()) {
            if (LocallyVisited.find(User) == LocallyVisited.end()) {
              // if (Visited.find(User) != Visited.end()) {
              Descendents.push(User);
              // errs() << "-----PUSHING\n";
              // User->dump();
              // }
            }
          }
        }
      /* } */
    }
    errs() << "END\n";
  }

}

bool CudaAnalysis::isFormedFromTerminalsOnly(Value *value) {
  bool onlyTerminals = true;
  std::stack<Value*> Stack;
  std::set<Value*> Visited;
  errs() << "Checking if only terminals are used\n";
  value->dump();

  Stack.push(value);

  while(!Stack.empty()) {
    Value *Current = Stack.top();
    Current->dump();
    Stack.pop();
    if(Visited.find(Current) != Visited.end()) {
      errs() << "hi\n";
      continue;
    }
    if(!isa<Instruction>(Current)){
      if(TerminalValues.find(Current)==TerminalValues.end() && !isa<ConstantInt>(Current)){
        onlyTerminals = false;
        break;
      }
    }
    if (isa<Instruction>(Current)) {
      auto *In = dyn_cast<Instruction>(Current);
      for (auto &Operand : In->operands()) {
        Stack.push(Operand);
      }
      Visited.insert(Current);
      continue;
    }
  }

  errs() << "hill " << onlyTerminals << "\n";
  return onlyTerminals;

}

void CudaAnalysis::analyzeGEPIndex(Value *MemInst, Value *Index) {
  // errs() << "Analyzing GEPI\n";
  // Index->dump();
  std::stack<Value *> ValueQueue;
  std::set<Value *> Visited;
  std::set<Value *> AddOps;
  std::set<Value *>
      Terms; // we store the operand of adds which form the top level terms

  // Now we perform a traversal of the GEP index and find out dependencies;
  ValueQueue.push(Index);
  Value *Top;
  while (!ValueQueue.empty()) {
    Top = ValueQueue.top();
    // Top->dump();
    ValueQueue.pop();
    Visited.insert(Top);
    if (auto *In = dyn_cast<Instruction>(Top)) {
      // if (auto *CI = dyn_cast<CallInst>(In)) {
      // }
      if (SpecialValues.find(In) != SpecialValues.end()) {
        // errs() << "special value \n";
        // In->dump();
      } else {
        for (auto &Operand : In->operands()) {
          // Operand->dump();
          if (Visited.find(Operand) == Visited.end()) {
            if (isa<Instruction>(Operand)) {
              ValueQueue.push(Operand);
            }
          }
        }
      }
    }
  }

  assert(ValueQueue.empty()); // must be emtpy by now.
  Visited.clear();

  // errs() << "\nCHECKING for terms\n";

  // Now we perform a traversal of the GEP index and find out dependencies;
  ValueQueue.push(Index);
  // Value *Top;
  while (!ValueQueue.empty()) {
    Top = ValueQueue.top();
    // Top->dump();
    ValueQueue.pop();
    Visited.insert(Top);
    if (auto *In = dyn_cast<Instruction>(Top)) {
      if (isa<BinaryOperator>(In)) {
        if (In->getOpcode() == Instruction::Add) {
          // errs() << "ADDITION OP\n";
          AddOps.insert(In);
          for (auto &Operand : In->operands()) {
            // Operand->dump();
            if (Visited.find(Operand) == Visited.end()) {
              if (isa<Instruction>(Operand)) {
                ValueQueue.push(Operand);
              }
            }
          }
        } else if ((In->getOpcode() == Instruction::Mul) ||
                   (In->getOpcode() == Instruction::Shl)) {
          errs() << "multiply op\n";
          In->dump();
          if (IndexSubComputationToMultiplierMap.find(In) ==
              IndexSubComputationToMultiplierMap.end()) {
            auto *Oper0 = In->getOperand(0);
            auto *Oper1 = In->getOperand(1);
            Oper0->dump();
            Oper1->dump();
            // change to only DIM values
            if ((GridDimValues.find(Oper0) != GridDimValues.end()) ||
                isa<ConstantInt>(Oper0) || isFormedFromTerminalsOnly(Oper0) ) { // add a condition to check if only formed from terminals
              IndexSubComputationToMultiplierMap[In].push_back(Oper0);
              errs() << "added\n";
            }
            if ((GridDimValues.find(Oper1) != GridDimValues.end()) ||
                isa<ConstantInt>(Oper1) || isFormedFromTerminalsOnly(Oper1) ) {
              IndexSubComputationToMultiplierMap[In].push_back(Oper1);
              errs() << "added\n";
            }
          }
        }
      } else { // not a binary operator
        for (auto &Operand : In->operands()) { // Operand->dump();
          if (Visited.find(Operand) == Visited.end()) {
            if (isa<Instruction>(Operand)) {
              ValueQueue.push(Operand);
            }
          }
        }
      }
    }
  }

  errs() << "\nPRINTING OUT INDEX SUB COMPUTATION MULTIPLIERS\n";
  for (auto ValueIter = IndexSubComputationToMultiplierMap.begin();
       ValueIter != IndexSubComputationToMultiplierMap.end(); ValueIter++) {
    (*ValueIter).first->dump();
    auto MulVector = (*ValueIter).second;
    for (auto MulIter = MulVector.begin(); MulIter != MulVector.end();
         MulIter++) {
      errs() << "   ";
      (*MulIter)->dump();
    }
  }

  errs() << "CHECKINg FOR TERMS OVER\n";
  for (auto AddIter = AddOps.begin(); AddIter != AddOps.end(); AddIter++) {
    // (*AddIter)->dump();
    auto *AddOp = dyn_cast<BinaryOperator>(*AddIter);
    for (auto &Operand : AddOp->operands()) {
      if (AddOps.find(Operand) == AddOps.end()) {
        Terms.insert(Operand);
      }
    }
  }

  // errs() << "PRINTING FOR TERMS \n";
  // for (auto TermIter = Terms.begin(); TermIter != Terms.end(); TermIter++) {
  //   (*TermIter)->dump();
  // }
  // errs() << "PRINTING FOR TERMS OVER\n\n";

  MemoryOpToStrideInfoMap[MemInst] =
      new TermAnalysisSupportStruct(AddOps, Terms);
  return;
}

AxisValueType getAxisValueType(Value *V) { return AXIS_TYPE_NONE; }

SpecialValueType getSpecialRegisterReadType(CallInst *CI) {
  auto *Callee = CI->getCalledFunction();
  assert(Callee && Callee->getName().startswith("llvm.nvvm.read.ptx.sreg"));
  // errs() << Callee->getName();
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
    return SREG_TIDX;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.y") {
    return SREG_TIDY;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.z") {
    return SREG_TIDZ;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.ctaid.x") {
    return SREG_BIDX;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.ctaid.y") {
    return SREG_BIDY;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.ctaid.z") {
    return SREG_BIDZ;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.ntid.x") {
    return SREG_BDIMX;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.ntid.y") {
    return SREG_BDIMY;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.ntid.z") {
    return SREG_BDIMZ;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.nctaid.x") {
    return SREG_GDIMX;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.nctaid.y") {
    return SREG_GDIMY;
  }
  if (Callee->getName() == "llvm.nvvm.read.ptx.sreg.nctaid.z") {
    return SREG_GDIMZ;
  }
  return SREG_NONE;
}

bool CudaAnalysis::findSpecialValues(Function &F) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        auto *Callee = CI->getCalledFunction();
        if (Callee && Callee->getName().startswith("llvm.nvvm.read.ptx.sreg")) {
          SpecialValueType SRegType = getSpecialRegisterReadType(CI);
          assert(SRegType != SREG_NONE);
          SpecialValues[&I] = SRegType;
          TerminalValues.insert(&I);
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.ntid")) {
          SpecialValueType SRegType = getSpecialRegisterReadType(CI);
          assert(SRegType != SREG_NONE);
          GridDimValues[&I] = SRegType;
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.ntid.x")) {
          AxisValues[&I] = AXIS_TYPE_BDIMX;
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.ntid.y")) {
          AxisValues[&I] = AXIS_TYPE_BDIMY;
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.ctaid.x")) {
          AxisValues[&I] = AXIS_TYPE_BIDX;
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.ctaid.y")) {
          AxisValues[&I] = AXIS_TYPE_BIDY;
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.tid.x")) {
          AxisValues[&I] = AXIS_TYPE_TIDX;
        }
        if (Callee &&
            Callee->getName().startswith("llvm.nvvm.read.ptx.sreg.tid.y")) {
          AxisValues[&I] = AXIS_TYPE_TIDY;
        }
        // CI->dump();
        // SpecialValues[&I] = std::string("SpecialRegisterRead");
      }
    }
  }
  // for (auto I = GridDimValues.begin(); I != GridDimValues.end(); I++) {
  //   I->first->dump();
  //   errs() << I->second << "\n\n";
  // }

  // // errs() << "Pattern matching for known pattersn\n\n";
  // for (auto &BB : F) {
  //   for (auto &I : BB) {
  //     if (I.isBinaryOp()) {
  //       auto *Oper0 = dyn_cast<CallInst>(I.getOperand(0));
  //       auto *Oper1 = dyn_cast<CallInst>(I.getOperand(1));
  //       auto CompoundSpecial0 = SpecialValues.find(I.getOperand(0));
  //       auto CompoundSpecial1 = SpecialValues.find(I.getOperand(1));
  //       if (Oper0 && Oper1) {
  //         if (isSpecialRegisterRead(Oper0) && isSpecialRegisterRead(Oper1)) {
  //           errs() << "SPL REG READ MIX OP\n";
  //           // I.dump();
  //           // I.getOperand(0)->dump();
  //           // I.getOperand(1)->dump();
  //           // errs() << "\n";
  //           SpecialValues[&I] = SREG_COMPOUND;
  //         }
  //       } else {
  //         if (Oper0 && isSpecialRegisterRead(Oper0) &&
  //             CompoundSpecial1 != SpecialValues.end()) {
  //           // errs() << "SPL REG READ MIX OP\n";
  //           // I.dump();
  //           // I.getOperand(0)->dump();
  //           // I.getOperand(1)->dump();
  //           // errs() << "\n";
  //           SpecialValues[&I] = SREG_COMPOUND;
  //         }
  //         if (Oper1 && isSpecialRegisterRead(Oper1) &&
  //             CompoundSpecial0 != SpecialValues.end()) {
  //           // errs() << "SPL REG READ MIX OP\n";
  //           // I.dump();
  //           // I.getOperand(0)->dump();
  //           // I.getOperand(1)->dump();
  //           // errs() << "\n";
  //           SpecialValues[&I] = SREG_COMPOUND;
  //         }
  //         if (CompoundSpecial0 != SpecialValues.end() &&
  //             CompoundSpecial1 != SpecialValues.end()) {
  //           // errs() << "SPL REG READ MIX OP\n";
  //           // I.dump();
  //           // I.getOperand(0)->dump();
  //           // I.getOperand(1)->dump();
  //           // errs() << "\n";
  //           SpecialValues[&I] = SREG_COMPOUND;
  //         }
  //       }
  //     }
  //   }
  // }

  // errs() << "Special values after compund check\n";
  // for (auto I = SpecialValues.begin(); I != SpecialValues.end(); I++) {
  //   I->first->dump();
  //   errs() << I->second << "\n\n";
  // }
  return false;
}

void CudaAnalysis::printTreeForStrideComputation(Value *V) {
  std::stack<Value *> ValueQueue;
  std::set<Value *> Visited;
  // Now we perform a traversal of the GEP index and find out dependencies;
  ValueQueue.push(V);
  Value *Top;
  while (!ValueQueue.empty()) {
    Top = ValueQueue.top();
    errs() << " .  . ";
    Top->dump();
    ValueQueue.pop();
    Visited.insert(Top);
    if (auto *In = dyn_cast<Instruction>(Top)) {
      // if (auto *CI = dyn_cast<CallInst>(In)) {
      // }
      if (SpecialValues.find(In) != SpecialValues.end()) {
        errs() << "--->> special value \n";
        // In->dump();
      } else if (LoopInductionVariables.find(In) !=
                 LoopInductionVariables.end()) {
        errs() << "--->> loop induction value \n";
        // In->dump();
      } else {
        for (auto &Operand : In->operands()) {
          // Operand->dump();
          if (Visited.find(Operand) == Visited.end()) {
            if (isa<Instruction>(Operand)) {
              ValueQueue.push(Operand);
            }
          }
        }
      }
    }
  }
  return;
}

bool CudaAnalysis::computeStrides() {
  errs() << "PRINTING TERM INFO\n";
  for (auto MemInstIter = MemoryOpToStrideInfoMap.begin();
       MemInstIter != MemoryOpToStrideInfoMap.end(); MemInstIter++) {
    errs() << "Mem Inst\n";
    (*MemInstIter).first->dump();
    auto *MemInst = (*MemInstIter).first;
    auto *TermInfoStruct = (*MemInstIter).second;
    errs() << "Terms\n";
    for (auto TermIter = TermInfoStruct->Terms.begin();
         TermIter != TermInfoStruct->Terms.end(); TermIter++) {
      (*TermIter)->dump();
      printTreeForStrideComputation(*TermIter);
    }
  }
  return false;
}

void CudaAnalysis::computeReuse(LoopInfo &LI, Function &F,
                                std::vector<Value *> KernelArgVector) {
  errs() << "COMPUTING REUSE\n";
  std::map<Value *, SpecialValueType> ReuseForMemoryAllocation;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LdI = dyn_cast<LoadInst>(&I)) {
        for (Use &U : LdI->operands()) {
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "\nFOUND GEP\n";
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "\nFOUND GLOBAL MEMROY GEP\n";
              // LdI->dump();
              // G->dump();
              analyzeGEPIndexForReuse(&I, G, G->getOperand(1));
            }
          }
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        for (Use &U : SI->operands()) {
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "\nFOUND GEP\n";
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "\nFOUND GLOBAL MEMROY GEP\n";
              // SI->dump();
              // G->dump();
              analyzeGEPIndexForReuse(&I, G, G->getOperand(1));
            }
          }
        }
      }
    }
  }

  // errs() << "\n\nPRINTING Memory GEPI to Multiplier Vector Map\n";
  // for (auto MGMVMIter = MemoryOpToMuliplierVectorMap.begin();
  //      MGMVMIter != MemoryOpToMuliplierVectorMap.end(); MGMVMIter++) {
  //   Value *MemOp = (*MGMVMIter).first;
  //   MemOp->dump();
  //   if (auto *SI = dyn_cast<StoreInst>(MemOp)) {
  //     SI->getPointerOperand()->dump();
  //   } else if (auto *LdI = dyn_cast<LoadInst>(MemOp)) {
  //     LdI->getPointerOperand()->dump();
  //   }
  //   errs() << "\n";
  //   std::map<Value *, std::vector<Value *>> Axes = (*MGMVMIter).second;
  //   for (auto AxisIter = Axes.begin(); AxisIter != Axes.end(); AxisIter++) {
  //     (*AxisIter).first->dump();
  //     std::vector<Value *> Multipliers = (*AxisIter).second;
  //     for (auto MulIter = Multipliers.begin(); MulIter != Multipliers.end();
  //          MulIter++) {
  //       (*MulIter)->dump();
  //     }
  //     errs() << "\n";
  //   }
  //   errs() << "\n";
  //   errs() << "\n";
  // }

  errs() << "\nWRITING TO REUSE FILE\n\n";
  if (!ReuseDetailfile.is_open()) {
    errs() << "Unable to open file\n";
  } else {
    for (auto MGMVMIter = MemoryOpToMuliplierVectorMap.begin();
        MGMVMIter != MemoryOpToMuliplierVectorMap.end(); MGMVMIter++) {
      Value *MemOp = (*MGMVMIter).first;
      MemOp->dump();
      Value *GEPI = nullptr;
      if (auto *SI = dyn_cast<StoreInst>(MemOp)) {
        GEPI = SI->getPointerOperand();
      } else if (auto *LdI = dyn_cast<LoadInst>(MemOp)) {
        GEPI = LdI->getPointerOperand();
      }
      // Handle direct pointer cases
      if (auto *GEP = dyn_cast<GetElementPtrInst>(GEPI)) {
        Value *Ptr = GEP->getPointerOperand();
        auto It =
          std::find(KernelArgVector.begin(), KernelArgVector.end(), Ptr);
        std::map<Value *, std::vector<Value *>> Axes = (*MGMVMIter).second;
        for (auto AxisIter = Axes.begin(); AxisIter != Axes.end(); AxisIter++) {
          // (*AxisIter).first->dump();
          ReuseDetailfile << F.getName().str() << " "
            << It - KernelArgVector.begin();
          ReuseDetailfile << " " << MemoryOpToAccessIDMap[MemOp];
          ReuseDetailfile << " " << MemoryOpToNumAccessMap[MemOp];
          (*AxisIter).first->dump();
          auto AxIter = AxisValues.find((*AxisIter).first);
          if(AxIter != AxisValues.end()) {
            ReuseDetailfile << " "
              << AxisValueNames[AxisValues[(*AxisIter).first]];
          } 
          auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), (*AxisIter).first);
          if(It != KernelArgVector.end()) {
            std::string arg = "ARG";
            auto argid = It - KernelArgVector.begin();
            arg.append(std::to_string(argid));
            ReuseDetailfile << " "
              << arg;
          }

          /* errs() << " axis = " << AxisValueNames[AxisValues[(*AxisIter).first]] << "\n"; */
          std::vector<Value *> Multipliers = (*AxisIter).second;
          errs() << "multipliers\n";
          for (auto MulIter = Multipliers.begin(); MulIter != Multipliers.end();
              MulIter++) {
            // TODO : for differnt possible types of multipiers (constants,
            // BIDs), find out the relavant value and print it out
            ReuseDetailfile << " [" << getMultiplierString(*MulIter) << "] ";
            (*MulIter)->dump();
          }
          ReuseDetailfile << "\n";
        }
      }
    }
  }
}

bool CudaAnalysis::findDirectionOfAccess(LoopInfo &LI, Function &F,
                                         std::vector<Value *> KernelArgVector) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LdI = dyn_cast<LoadInst>(&I)) {
        // errs() << "\n";
        // LdI->dump();
        for (Use &U : LdI->operands()) {
          // U->dump();
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "\nFOUND GEP\n";
            // G->dump();
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "pointer address space = "
              //  << G->getPointerAddressSpace() << "\n";
              // errs() << "Pointer Operand\n";
              // G->getPointerOperand()->dump();
              // errs() << "Index Operand\n";
              // NOTE: THIS IS VERY SPECIFIC to our test case.
              // GEP has a million combinations.
              // G->getOperand(1) ->dump();
              analyzeGEPIndex(&I, G->getOperand(1));
            }
          }
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        // SI->dump();
        for (Use &U : SI->operands()) {
          // U->dump();
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "\nFOUND GEP\n";
            // G->dump();
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "pointer address space = "
              //  << G->getPointerAddressSpace() << "\n";
              // errs() << "Pointer Operand\n";
              // G->getPointerOperand()->dump();
              // errs() << "Index Operand\n";
              // NOTE: THIS IS VERY SPECIFIC to our test case.
              // GEP has a million combinations.
              // G->getOperand(1) ->dump();
              analyzeGEPIndex(&I, G->getOperand(1));
            }
          }
        }
      }
    }
  }
  return false;
}

Value* CudaAnalysis::getIndirectMemop(Value* V){
  // iterate through GEP chain till reaching args
  errs() << "indirect memop: ";
  V->dump();
  if (auto *LdI = dyn_cast<LoadInst>(V)) {
    Value* PO = LdI->getPointerOperand();
    auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), PO);
    if(It == KernelArgVector.end()) {
      errs() << "recurse\n";
      getIndirectMemop(PO);
    } else {
      errs() << "return\n";
      return PO;
    }
  } else if (auto * GEPI = dyn_cast<GetElementPtrInst>(V)) {
    Value* PO = GEPI->getPointerOperand();
    auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), PO);
    if(It == KernelArgVector.end()) {
      getIndirectMemop(PO);
    } else {
      return PO;
    }
  }
}

bool CudaAnalysis::countMemoryOperations(LoopInfo &LI, Function &F,
    std::vector<Value *> KernelArgVector) {
  errs() << "Count memory operations\n";
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LdI = dyn_cast<LoadInst>(&I)) {
        errs() << "----\n";
        LdI->dump();
        auto U = LdI->getPointerOperand();
        // errs() << "..";
        // U->dump();
        // if (auto G = dyn_cast<GetElementPtrInst>(U)) {
        if (auto *G = recursiveFindPointer(U)) {
          // errs() << "found GEP\n";
          unsigned Iters;
          if (Loop *Loop = LI.getLoopFor(LdI->getParent())) {
            Iters = LoopToTotalIterMapping[Loop];
          } else {
            Iters = 1;
          }
          auto *Ptr = G;
          if (!isSharedMemoryAccess(G)) {
            // errs() << "pointer address space = "
            //        << G->getPointerAddressSpace() << "\n";
            errs() << "\n\n";
            LdI->dump();
            G->dump();
            if (PointerInfoMap.find(Ptr) == PointerInfoMap.end()) {
              errs() << "NOT FOUND. NEED TO DO RECURSIVE\n";
              // if a memory access is to a stack variable, then it doesn't count.
              if(ByValArgsSet.find(Ptr) != ByValArgsSet.end()){
                errs() << "found by value (arg maybe a struct) \n";
                StackAccesses.insert(LdI);
                Ptr->dump();
              }
              if(StackAccesses.find(Ptr) != StackAccesses.end()) {
                errs() << "accssing an address from stack (not on stack)\n";
                Ptr->dump();
                LoadInst* SecondLoad = dyn_cast<LoadInst>(Ptr);
                Value* ByValArg = SecondLoad->getPointerOperand();
                errs() << "by val arg\n";
                ByValArg->dump();
                if(ByValArgsSet.find(ByValArg) != ByValArgsSet.end()){
                  errs() << "found the struct \n";
                  auto Arg = dyn_cast<Argument>(ByValArg);
                  Arg->getParamByValType()->dump();
                  MemoryOpToAccessIDMap[&I] = AccessID++; // post increment
                  if (Loop *Loop = LI.getLoopFor(LdI->getParent())) {
                    MemoryOpToEnclosingLoopMap[&I] = Loop;
                  } else {
                    errs() << "Unable to find enclosing loop\n";
                  }
                  MemoryOpToPointerMap[&I] = Ptr;
                  isInsideConditional(&I);
                }
              }
            } else {
              PointerInfoMap[Ptr]->Loads += Iters;
              MemoryOpToNumAccessMap[&I] = Iters;
              MemoryOpToAccessIDMap[&I] = AccessID++; // post increment
              if (Loop *Loop = LI.getLoopFor(LdI->getParent())) {
                MemoryOpToEnclosingLoopMap[&I] = Loop;
              } else {
                errs() << "Unable to find enclosing loop\n";
              }
              MemoryOpToPointerMap[&I] = Ptr;
                  isInsideConditional(&I);
            }
          } else {
            errs() << "shared memory access\n";
          }
        }
        else {
          errs() << "Unable to find pointer\n";
        }
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        errs() << "----\n";
        SI->dump();
        // if (auto G = dyn_cast<GetElementPtrInst>(U)) {
        auto U = SI->getPointerOperand();
        // errs() << "found GEP\n";
        if(auto *G = recursiveFindPointer(U)){
          unsigned Iters;
          if (Loop *Loop = LI.getLoopFor(SI->getParent())) {
            Iters = LoopToTotalIterMapping[Loop];
            // errs() << "parent loop found\n";
          } else {
            Iters = 1;
            // errs() << "NOT FOUND \n";
          }
          auto *Ptr = G;
          errs() << "GEP\n";
          G->dump();
          if (!isSharedMemoryAccess(G)) {
            // errs() << "pointer address space = "
            //  << G->getPointerAddressSpace() << "\n";
            errs() << "\n";
            SI->dump();
            G->dump();
            if (PointerInfoMap.find(Ptr) == PointerInfoMap.end()) {
              errs() << "NOT FOUND. NEED TO DO RECURSIVE\n";
              // if a memory access is to a stack variable, then it doesn't count.
              /* if(ByValArgsSet.find(Ptr) != ByValArgsSet.end()){ */
              /*   errs() << "found by value (arg maybe a struct) \n"; */
              /*   StackAccesses.insert(SI); */
              /*   Ptr->dump(); */
              /* } */
              if(StackAccesses.find(Ptr) != StackAccesses.end()) {
                errs() << "accssing an address from stack (not on stack)\n";
                Ptr->dump();
                LoadInst* SecondLoad = dyn_cast<LoadInst>(Ptr);
                Value* ByValArg = SecondLoad->getPointerOperand();
                errs() << "by val arg\n";
                ByValArg->dump();
                if(ByValArgsSet.find(ByValArg) != ByValArgsSet.end()){
                  errs() << "found the struct \n";
                  auto Arg = dyn_cast<Argument>(ByValArg);
                  Arg->getParamByValType()->dump();
                  MemoryOpToAccessIDMap[&I] = AccessID++; // post increment
                  if (Loop *Loop = LI.getLoopFor(SI->getParent())) {
                    MemoryOpToEnclosingLoopMap[&I] = Loop;
                  } else {
                    errs() << "Unable to find enclosing loop\n";
                  }
                  MemoryOpToPointerMap[&I] = Ptr;
                  isInsideConditional(&I);
                }
              }
            } else {
              errs() << "found in pointer map\n";
              PointerInfoMap[Ptr]->Loads += Iters;
              MemoryOpToNumAccessMap[&I] = Iters;
              MemoryOpToAccessIDMap[&I] = AccessID++; // post increment
              if (Loop *Loop = LI.getLoopFor(SI->getParent())) {
                MemoryOpToEnclosingLoopMap[&I] = Loop;
              } else {
                // MemoryOpToEnclosingLoopMap[&I] = nullptr;
                errs() << "Unable to find enclosing loop\n";
              }
              MemoryOpToPointerMap[&I] = Ptr;
                  isInsideConditional(&I);
            }
          } else {
          }
        } else {
        }
      }
      }
      }

      errs() << "Attempting file write\n";
      if (!AccessDetailFile.is_open()) {
        errs() << "Unable to open file\n";
      } else {
        for (auto I = MemoryOpToPointerMap.begin(); I != MemoryOpToPointerMap.end();
            I++) {
          errs() << "memory op\n";
          I->first->dump();
          I->second->dump();
          // auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(),
          // I->second);

          AccessDetailFile << F.getName().str() << " ";
          AccessDetailFile << MemoryOpToAccessIDMap[I->first] << " ";
          auto It =
            std::find(KernelArgVector.begin(), KernelArgVector.end(), I->second);
          errs() << MemoryOpToAccessIDMap[I->first] << " " << It - KernelArgVector.begin() << " ";
          if(It != KernelArgVector.end()){
            AccessDetailFile << It - KernelArgVector.begin() << " ";
          } else {
            // TODO: indirect search
            Value* arg = getIndirectMemop(I->second);
          auto It = std::find(KernelArgVector.begin(), KernelArgVector.end(), arg);
            AccessDetailFile << It - KernelArgVector.begin() << " ";
          }

          if (MemoryOpToEnclosingLoopMap.find(I->first) !=
              MemoryOpToEnclosingLoopMap.end()) {
            MemoryOpToEnclosingLoopMap[I->first]->dump();
            AccessDetailFile << LoopToLoopIdMapping[MemoryOpToEnclosingLoopMap[I->first]] << " ";
          } else {
            errs() << "no enclosing loop\n";
            AccessDetailFile << "0 ";
          }
          errs() << "\n";

          // print if 
          if (MemoryOpToIfBranch.find(I->first) !=
              MemoryOpToIfBranch.end()) {
            errs() << "inside if\n";
            MemoryOpToIfBranch[I->first]->dump();
            auto br = MemoryOpToIfBranch[I->first];
            errs() << BranchToBranchIdMapping[br] << "\n";
            AccessDetailFile << BranchToBranchIdMapping[br] << " ";
            if(MemoryOpToIfType[I->first] == true) {
                AccessDetailFile << " 1 ";
                errs() << "adf true\n";
            } else {
                AccessDetailFile << " 0 ";
                errs() << "adf false\n";
            }
          } else {
            AccessDetailFile << " 0 ";
            AccessDetailFile << " 0 ";
          }

          bool isPtrChase = isPointerChaseFixed(I->first);
          auto expression = getExpressionTree(I->first);
          auto expr_strings = convertValuesToStrings(expression);
          errs() << "expression strings\n";
          AccessDetailFile << "[ ";
          if(!isPtrChase){
            for (auto ArgIter = expr_strings.begin(); ArgIter != expr_strings.end(); ArgIter++){
              errs() << (*ArgIter) << "  ";
              AccessDetailFile << (*ArgIter) << "  ";
            }
            errs() << " \n";
          } else {
            AccessDetailFile << "PC";
          }
            AccessDetailFile << " ]";

          AccessDetailFile << "\n";


          // Access tree file
          AccessTreeFile << F.getName().str() << " ";
          AccessTreeFile << MemoryOpToAccessIDMap[I->first] << " ";
          std::ostringstream serializedExprTree;
          isPtrChase = isPointerChaseFixed(I->first);
          if(!isPtrChase){
              std::set<Value*> PhiNodesVisited;
              errs() << "hihi serialize expr tree\n";
              serializeExpressionTree(I->first, serializedExprTree, PhiNodesVisited);
              AccessTreeFile << serializedExprTree.str() ;
              AccessTreeFile << "\n";
          } else {
              AccessTreeFile << "PC \n";
          }

        }
      }

      errs() << "writing if else information\n";
      for(auto br = BranchToBranchIdMapping.begin();
          br != BranchToBranchIdMapping.end(); br++) {
        if ((BranchProcessed[br->first]) == true ) {
          continue;
        }
        BranchProcessed[br->first] = true;
        errs() << "\nbr id = " << (br->second) << "\n";
        br->first->dump();
        IfDetailFile << br->second << " " ;
        Instruction* bri = dyn_cast<Instruction>(br->first);
        assert(bri);
        bri->getOperand(0)->dump();
        auto pred = bri->getOperand(0);
        auto expression = getExpressionTree(pred);
        /* IfDetailFile << "[ "; */
        auto expr_strings = convertValuesToStrings(expression);
          errs() << "expression strings\n";
            for (auto ArgIter = expr_strings.begin(); ArgIter != expr_strings.end(); ArgIter++){
              errs() << (*ArgIter) << "  ";
              IfDetailFile << (*ArgIter) << "  ";
            }
        /* IfDetailFile << " ]"; */
        IfDetailFile << "\n";
      }

      errs() << "writing loop to phi information\n";
      for(auto phi = PhiNodeToUIDMap.begin();
              phi != PhiNodeToUIDMap.end(); phi++) {
          /* Loop* loop = LoopInductionVariableToLoopMap[phi->first]; */
          /* auto loopid = LoopToLoopIdMapping[loop]; */
          //get parent of phi node if it is a loop
          
          auto loopid = 0;

          if (Loop *loop = LI.getLoopFor(phi->first->getParent())) {
              loopid = LoopToLoopIdMapping[loop];
          }
          PhiToLoopFile << phi->second << "  " << loopid  << "\n";
      }

      return false;
    }

bool CudaAnalysis::findNumberOfAccesses(LoopInfo &LI, Function &F,
    std::vector<Value *> KernelArgVector) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LdI = dyn_cast<LoadInst>(&I)) {
        // errs() << "\n";
        // LdI->dump();
        for (Use &U : LdI->operands()) {
          // errs() << "..";
          // U->dump();
          // if (auto G = dyn_cast<GetElementPtrInst>(U)) {
          if (auto *G = recursiveFindGep(U)) {
            // errs() << "found GEP\n";
            unsigned Iters;
            if (Loop *Loop = LI.getLoopFor(LdI->getParent())) {
              Iters = LoopToTotalIterMapping[Loop];
            } else {
              Iters = 1;
            }
            auto *Ptr = G->getPointerOperand();
            if (!isSharedMemoryAccess(G->getPointerOperand())) {
              // errs() << "pointer address space = "
              //        << G->getPointerAddressSpace() << "\n";
              errs() << "\n";
              LdI->dump();
              G->getPointerOperand()->dump();
              if (PointerInfoMap.find(Ptr) == PointerInfoMap.end()) {
                errs() << "NOT FOUND. NEED TO DO RECURSIVE\n";
              } else {
                PointerInfoMap[Ptr]->Loads += Iters;
                MemoryOpToNumAccessMap[&I] = Iters;
                MemoryOpToAccessIDMap[&I] = AccessID++; // post increment
              }
            } else {
              // errs() << "shared memory access\n";
            }
          } else {
            // errs() << "GEP not found\n";
          }
        }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          // SI->dump();
          for (Use &U : SI->operands()) {
            // U->dump();
            // if (auto G = dyn_cast<GetElementPtrInst>(U)) {
            if (auto *G = recursiveFindGep(U)) {
              // errs() << "found GEP\n";
              unsigned Iters;
              if (Loop *Loop = LI.getLoopFor(SI->getParent())) {
                Iters = LoopToTotalIterMapping[Loop];
                // errs() << "parent loop found\n";
              } else {
                Iters = 1;
                // errs() << "NOT FOUND \n";
              }
              auto *Ptr = G->getPointerOperand();
              if (!isSharedMemoryAccess(G->getPointerOperand())) {
                // errs() << "pointer address space = "
                //  << G->getPointerAddressSpace() << "\n";
                errs() << "\n";
                SI->dump();
                G->getPointerOperand()->dump();
                if (PointerInfoMap.find(Ptr) == PointerInfoMap.end()) {
                  errs() << "NOT FOUND. NEED TO DO RECURSIVE\n";
                } else{
                  PointerInfoMap[Ptr]->Loads += Iters;
                  MemoryOpToNumAccessMap[&I] = Iters;
                  MemoryOpToAccessIDMap[&I] = AccessID++; // post increment
                }
              } else {
                // errs() << "shared memory access\n";
              }
            } else {
              // errs() << "GEP not found\n";
            }
          }
          }
        }
      }

      // errs() << "ARGUMENTS of " << &F << "\n";
      // for (auto ArgIter = KernelArgVector.begin(); ArgIter != KernelArgVector.end();
      //      ArgIter++) {
      //   (*ArgIter)->dump();
      // }

      // errs() << "\nCounts of #access to kernel parameter arrays\n";
      // for (auto I = PointerInfoMap.begin(); I != PointerInfoMap.end(); I++) {
      //   I->first->dump();
      //   auto It =
      //       std::find(KernelArgVector.begin(), KernelArgVector.end(), I->first);
      //   errs() << "Location in kernel arg list  " << It - KernelArgVector.begin()
      //          << "\n";
      //   errs() << "Count  " << I->second->Loads << "\n";
      // }

      errs() << "Attempting file write\n";
      if (!AccessDetailFile.is_open()) {
        errs() << "Unable to open file\n";
      } else {
        for (auto I = PointerInfoMap.begin(); I != PointerInfoMap.end(); I++) {
          // I->first->dump();
          auto It =
            std::find(KernelArgVector.begin(), KernelArgVector.end(), I->first);
          AccessDetailFile << F.getName().str() << " "
            << It - KernelArgVector.begin() << " "
            << I->second->Loads << "\n";
        }
      }
      return false;
    }

} // namespace

char CudaAnalysis::ID = 0;
static RegisterPass<CudaAnalysis> X("CudaAnalysis", "CudaAnalysis World Pass");
