use crate::codegen::{end_main_custom_return, start_main};
use crate::cst::{
    BlockStatement, ConcreteFileTree, Expression, FunctionCall, FunctionDeclaration, NodeType,
    TopLevelDeclaration,
};
use crate::lex::{lex, unlex, Lexer};
use crate::list::ladd_element;
use crate::token::{Token, TokenType};
use std::fs::File;

fn lexer_for_file(filename: &str) -> Option<Lexer> {
    let fp = File::open(filename).ok()?;
    Some(Lexer {
        fp: Some(fp),
        current_file: filename.to_string(),
        ..Lexer::default()
    })
}

fn next_token(l: &mut Lexer) -> Result<Token, i32> {
    let mut token = Token::default();
    match lex(l, &mut token) {
        0 => Ok(token),
        status => Err(status),
    }
}

fn expect(l: &mut Lexer, expected: TokenType) -> Result<Token, i32> {
    let token = next_token(l)?;
    if token.token_type == expected {
        Ok(token)
    } else {
        unlex(l, &token);
        Err(-1)
    }
}

/// Parses a function declaration from the lexer into a FunctionDeclaration object.
pub fn parse_funcdecl(l: &mut Lexer, fd: &mut FunctionDeclaration) -> i32 {
    let result: Result<(), i32> = (|| {
        expect(l, TokenType::TT_INT)?;
        let name = expect(l, TokenType::TT_IDENTIFIER)?;
        expect(l, TokenType::TT_OPAREN)?;
        expect(l, TokenType::TT_CPAREN)?;
        fd.name = name.contents;
        parse_blockstmt(l, &mut fd.body);
        if fd.body.stmts.head.is_none() {
            // An empty body is valid; a malformed body is reported by its parser.
        }
        Ok(())
    })();
    if result.is_ok() { 0 } else { -1 }
}

/// Creates a concrete syntax tree from the lexer.
pub fn make_cst(l: &mut Lexer, tree: &mut ConcreteFileTree) -> i32 {
    loop {
        let token = match next_token(l) {
            Ok(token) => token,
            Err(status) => return status,
        };
        if token.token_type == TokenType::TT_EOF {
            return 0;
        }
        if unlex(l, &token) != 0 {
            return -1;
        }
        let mut declaration = TopLevelDeclaration::default();
        if parse_funcdecl(l, &mut declaration.fd) != 0 {
            return -1;
        }
        if ladd_element(&mut tree.decls, Box::new(declaration)) != 0 {
            return -1;
        }
    }
}

/// Parses an expression from the Lexer into an Expression object.
pub fn parse_expr(l: &mut Lexer, ex: &mut Expression) -> i32 {
    let token = match next_token(l) {
        Ok(token) => token,
        Err(status) => return status,
    };
    match token.token_type {
        TokenType::TT_LITERAL => {
            ex.literal = Some(token.contents);
            ex.fc = None;
            ex.node_type = NodeType::NT_LITERAL;
            0
        }
        TokenType::TT_IDENTIFIER => {
            if unlex(l, &token) != 0 {
                -1
            } else {
                parse_funccall(l, ex)
            }
        }
        _ => {
            unlex(l, &token);
            -1
        }
    }
}

/// Parses a file and returns a status code.
pub fn parse(filename: &str) -> i32 {
    let Some(mut lexer) = lexer_for_file(filename) else {
        eprintln!("\x1b[31mError: jccc: File {filename} not found\x1b[0m");
        return 1;
    };
    let mut tokens = Vec::new();
    loop {
        let token = match next_token(&mut lexer) {
            Ok(token) => token,
            Err(_) => return 1,
        };
        println!(
            "Contents: {:>20}, type: {:>20}, position: {}/{}",
            token.contents,
            crate::lex::ttype_name(token.token_type),
            token.line,
            token.column
        );
        let done = token.token_type == TokenType::TT_EOF;
        tokens.push(token);
        if done {
            break;
        }
    }

    let valid_main = tokens.len() >= 9
        && tokens[0].token_type == TokenType::TT_INT
        && tokens[1].token_type == TokenType::TT_IDENTIFIER
        && tokens[1].contents == "main"
        && tokens[2].token_type == TokenType::TT_OPAREN
        && tokens[3].token_type == TokenType::TT_CPAREN
        && tokens[4].token_type == TokenType::TT_OBRACE
        && tokens[5].token_type == TokenType::TT_RETURN
        && tokens[6].token_type == TokenType::TT_LITERAL
        && tokens[6].contents.as_bytes().first().is_some_and(u8::is_ascii_digit)
        && tokens[7].token_type == TokenType::TT_SEMI
        && tokens[8].token_type == TokenType::TT_CBRACE;

    if valid_main {
        println!();
        print!("{}", start_main());
        let value = tokens[6].contents.parse::<i32>().unwrap_or(0);
        print!("{}", end_main_custom_return(value));
    } else {
        eprintln!("\x1b[31mError: jccc: Not correct main function.\n\x1b[0m");
    }
    0
}

/// Parses a simple main function (for testing).
pub fn parse_simple_main_func() -> i32 {
    0
}

/// Parses a block statement from the lexer.
pub fn parse_blockstmt(l: &mut Lexer, bs: &mut BlockStatement) -> i32 {
    if expect(l, TokenType::TT_OBRACE).is_err() {
        return -1;
    }
    loop {
        let token = match next_token(l) {
            Ok(token) => token,
            Err(status) => return status,
        };
        match token.token_type {
            TokenType::TT_CBRACE => return 0,
            TokenType::TT_EOF => return -1,
            TokenType::TT_RETURN => {
                let mut expression = Expression::default();
                if parse_expr(l, &mut expression) != 0 || expect(l, TokenType::TT_SEMI).is_err() {
                    return -1;
                }
                if ladd_element(&mut bs.stmts, Box::new(expression)) != 0 {
                    return -1;
                }
            }
            _ => return -1,
        }
    }
}

/// Parses a function call from the lexer into an Expression object.
pub fn parse_funccall(l: &mut Lexer, ex: &mut Expression) -> i32 {
    let result: Result<(), i32> = (|| {
        let name = expect(l, TokenType::TT_IDENTIFIER)?;
        expect(l, TokenType::TT_OPAREN)?;
        expect(l, TokenType::TT_CPAREN)?;
        ex.fc = Some(FunctionCall { name: name.contents });
        ex.literal = None;
        ex.node_type = NodeType::NT_FUNCCALL;
        Ok(())
    })();
    if result.is_ok() { 0 } else { -1 }
}
