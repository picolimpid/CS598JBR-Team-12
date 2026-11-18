use crate::list::{create_list, List};

/// Represents different node types in the AST/CST.
#[allow(non_camel_case_types)] // Names are fixed by the C syntax-tree interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    NT_STMT,
    NT_EXPR,
    NT_BLOCK_STMT,
    NT_RETURN_STMT,
    NT_FUNCDECL,
    NT_FUNCCALL,
    NT_LITERAL,
}

/// A block statement containing a list of statements.
#[derive(Debug)]
pub struct BlockStatement {
    pub stmts: List,
}

impl Default for BlockStatement {
    fn default() -> Self {
        Self {
            stmts: create_list(16),
        }
    }
}

/// A function declaration, including its body and name.
#[derive(Debug, Default)]
pub struct FunctionDeclaration {
    pub body: BlockStatement,
    pub name: String,
}

/// A top-level declaration node that holds a function declaration in this simplified design.
#[derive(Debug)]
pub struct TopLevelDeclaration {
    pub fd: FunctionDeclaration,
    pub node_type: NodeType,
}

impl Default for TopLevelDeclaration {
    fn default() -> Self {
        Self {
            fd: FunctionDeclaration::default(),
            node_type: NodeType::NT_FUNCDECL,
        }
    }
}

/// A function call node with a simple string name.
#[derive(Debug, Default)]
pub struct FunctionCall {
    pub name: String,
}

/// An expression node that can hold either a function call or a literal string.
#[derive(Debug)]
pub struct Expression {
    pub fc: Option<FunctionCall>,
    pub literal: Option<String>,
    pub node_type: NodeType,
}

impl Default for Expression {
    fn default() -> Self {
        Self {
            fc: None,
            literal: None,
            node_type: NodeType::NT_EXPR,
        }
    }
}

/// A file tree representing a list of top-level declarations.
#[derive(Debug)]
pub struct ConcreteFileTree {
    pub decls: List,
}

impl Default for ConcreteFileTree {
    fn default() -> Self {
        Self {
            decls: create_list(16),
        }
    }
}
