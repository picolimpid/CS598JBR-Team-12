use std::sync::Mutex;

/// An operation that can be applied to registers.
#[allow(non_camel_case_types)] // Operation names mirror the x86 C interface.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    OP_ADD,
    OP_SUB,
    OP_MOV,
    OP_NOP,
}

/// Represents the code generation state.
#[derive(Debug, Default)]
pub struct GenState {
    pub registers_in_use: u32,
    pub rsp_offset: u32,
}

static GEN_STATE: Mutex<GenState> = Mutex::new(GenState {
    registers_in_use: 0,
    rsp_offset: 0,
});

/// Generates code to operate on RAX with RDI according to the provided Op.
pub fn op_on_rax_with_rdi(op: Op) -> String {
    let name = match op {
        Op::OP_ADD => "add",
        Op::OP_SUB => "sub",
        Op::OP_MOV => "mov",
        Op::OP_NOP => "nop",
    };
    format!("\t{name} rax, rdi\n")
}

/// Initializes code for an integer literal.
pub fn init_int_literal(val: i32) -> String {
    let mut state = match GEN_STATE.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    state.rsp_offset = state.rsp_offset.wrapping_add(8);
    format!("\tmov [rsp+{}], {val}", state.rsp_offset)
}

/// Tests the function that initializes integer literals.
pub fn test_init_int_literal() -> i32 {
    println!("Running \"test_init_int_literal\"");
    code_gen_init();
    assert_eq!(init_int_literal(100), "\tmov [rsp+8], 100");
    0
}

/// Initializes the code generator.
pub fn code_gen_init() {
    let mut state = match GEN_STATE.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    *state = GenState::default();
}

/// Starts the main function (assembly or IR) generation.
pub fn start_main() -> String {
    "global _start\nsection .text\n\n_start:\n".to_string()
}

/// Starts a generic function definition.
pub fn start_func() -> String {
    "\tsub rsp, 32\tmov [rsp], r12\tmov [rsp+8], r13\tmov [rsp+16], r14\tmov [rsp+24], r15"
        .to_string()
}

/// Tests the operation on RAX with RDI.
pub fn test_op_on_rax_with_rdi() -> i32 {
    println!("Running \"test_op_on_rax_with_rdi\"");
    assert_eq!(op_on_rax_with_rdi(Op::OP_ADD), "\tadd rax, rdi\n");
    assert_eq!(op_on_rax_with_rdi(Op::OP_MOV), "\tmov rax, rdi\n");
    0
}

/// Ends a function definition (assembly or IR).
pub fn end_func() -> String {
    "\tmov r12, [rsp]\tmov r13, [rsp+8]\tmov r14, [rsp+16]\tmov r15, [rsp+24]\tadd rsp, 32"
        .to_string()
}

/// Ends the main function with a custom return value.
pub fn end_main_custom_return(val: i32) -> String {
    format!("\tmov rax, 60\n\tmov rdi, {val}\n\tsyscall\n")
}

/// Ends the main function with a default return.
pub fn end_main() -> String {
    "\tmov rax, 60\tmov rdi, 0\tsyscall".to_string()
}
