#![forbid(unsafe_code)]

use jccc::lex::{lex, ttype_name, Lexer};
use jccc::token::{Token, TokenType};
use std::fs::File;

/// Dumps lexer output for the specified file.
pub fn lexer_dump(filename: &str) -> i32 {
    let fp = match File::open(filename) {
        Ok(fp) => fp,
        Err(_) => {
            eprintln!("\x1b[31mError: jccc: File {filename} not found\x1b[0m");
            return 1;
        }
    };
    let mut lexer = Lexer {
        fp: Some(fp),
        current_file: filename.to_string(),
        ..Lexer::default()
    };

    loop {
        let mut token = Token::default();
        if lex(&mut lexer, &mut token) != 0 {
            return 1;
        }
        println!(
            "Contents: {:>20}, type: {:>20}, position: {}/{}",
            token.contents,
            ttype_name(token.token_type),
            token.line,
            token.column
        );
        if token.token_type == TokenType::TT_EOF {
            return 0;
        }
    }
}

/// The main entry point for the program.
pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match args.as_slice() {
        [] => {
            eprintln!("\x1b[37mjccc: Usage: --token-dump <filename> to see all tokens\x1b[0m");
            0
        }
        [filename] => {
            eprintln!("\x1b[37mjccc: default compilation not supported yet -- try 'jccc --token-dump {filename}' instead.\x1b[0m");
            1
        }
        [option, filename] if option == "--token-dump" => lexer_dump(filename),
        [option, filename] if option == "--test-parse" => jccc::parse::parse(filename),
        [_, _] => {
            eprintln!(
                "\x1b[31mError: jccc: option {} not recognized.\x1b[0m",
                args[1]
            );
            1
        }
        _ => {
            eprintln!("\x1b[37mjccc: expected only two arguments!\x1b[0m");
            1
        }
    };
    if status != 0 {
        std::process::exit(status);
    }
}
