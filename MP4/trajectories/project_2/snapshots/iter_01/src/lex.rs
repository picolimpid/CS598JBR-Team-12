use crate::token::{Token, TokenType, TOKEN_LENGTH};
use std::io::{Read, Seek, SeekFrom};

pub const TOKEN_PUTBACKS: usize = 5;

/// Represents a lexer for tokenizing input.
#[derive(Debug)]
pub struct Lexer {
    /// An optional file; in C this would be a FILE*, here we keep it as a File for demonstration.
    pub fp: Option<std::fs::File>,
    pub current_file: String,
    pub buffer: [u8; 1],
    pub position: i64,
    pub last_column: i32,
    pub column: i32,
    pub line: i32,
    pub unlexed: [Token; TOKEN_PUTBACKS],
    pub unlexed_count: u32,
}

impl Default for Lexer {
    fn default() -> Self {
        Self {
            fp: None,
            current_file: String::new(),
            buffer: [0],
            position: 0,
            last_column: 1,
            column: 1,
            line: 1,
            unlexed: std::array::from_fn(|_| Token::default()),
            unlexed_count: 0,
        }
    }
}

/// Gets the next character from the lexer.
pub fn lexer_getchar(l: &mut Lexer) -> i32 {
    l.position = l.position.wrapping_add(1);
    l.last_column = l.column;
    let mut byte = [0u8; 1];
    let result = match l.fp.as_mut() {
        Some(file) => match file.read(&mut byte) {
            Ok(0) | Err(_) => -1,
            Ok(_) => byte[0] as i8 as i32,
        },
        None => -1,
    };
    l.buffer[0] = result as u8;
    if result == b'\n' as i32 {
        l.line = l.line.wrapping_add(1);
        l.column = 0;
    } else {
        l.column = l.column.wrapping_add(1);
    }
    result
}

/// Un-gets (pushes back) the last character.
pub fn lexer_ungetchar(l: &mut Lexer) -> i32 {
    if l.position < 0 {
        return -1;
    }
    l.position = l.position.wrapping_sub(1);
    l.column = l.last_column;
    if l.buffer[0] == b'\n' {
        l.line = l.line.wrapping_sub(1);
    }
    if let Some(file) = l.fp.as_mut() {
        if file.seek(SeekFrom::Current(-1)).is_err() {
            return -1;
        }
    }
    1
}

/// Checks if character c is in the string s.
pub fn in_string(c: char, s: &str) -> i32 {
    i32::from(s.contains(c))
}

/// Checks if a character starts an operator sequence.
pub fn starts_operator(c: char) -> i32 {
    i32::from(matches!(
        c,
        '-' | '+' | '*' | '/' | '=' | ':' | '%' | '&' | '|' | '<' | '>' | '!' | '~' | '^'
    ))
}

/// Checks if the provided operator sequence is valid.
pub fn valid_operator_sequence(op: &str) -> i32 {
    const OPERATORS: &[&str] = &[
        "-", "+", "*", "/", "=", ":", "%", "&", "&&", "|", "||", "-=", "+=", "++",
        "--", "/=", "*=", "%=", "&=", "|=", "&&=", "||=", ">", "<", "<=", ">=", "<<",
        ">>", "!", "==", "!=", "^", "^=", "->", "<<=", ">>=",
    ];
    i32::from(OPERATORS.contains(&op))
}

/// Checks if c is a valid numeric or identifier character.
pub fn is_valid_numeric_or_id_char(c: char) -> i32 {
    i32::from(c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// Determines a token type from a single character.
pub fn ttype_one_char(c: char) -> TokenType {
    use TokenType::*;
    match c {
        '(' => TT_OPAREN,
        ')' => TT_CPAREN,
        '{' => TT_OBRACE,
        '}' => TT_CBRACE,
        '[' => TT_OBRACKET,
        ']' => TT_CBRACKET,
        ';' => TT_SEMI,
        '.' => TT_PERIOD,
        ',' => TT_COMMA,
        '-' => TT_MINUS,
        '+' => TT_PLUS,
        '*' => TT_STAR,
        '/' => TT_SLASH,
        '=' => TT_ASSIGN,
        ':' => TT_COLON,
        '%' => TT_MOD,
        '&' => TT_BAND,
        '|' => TT_BOR,
        '>' => TT_GREATER,
        '<' => TT_LESS,
        '!' => TT_LNOT,
        '~' => TT_BNOT,
        '^' => TT_XOR,
        '#' => TT_POUND,
        '?' => TT_QMARK,
        _ if c.is_ascii_digit() => TT_LITERAL,
        _ => TT_IDENTIFIER,
    }
}

/// Determines a token type from a multi-character sequence.
pub fn ttype_many_chars(contents: &str) -> TokenType {
    use TokenType::*;
    let keyword_or_operator = match contents {
        "auto" => TT_AUTO,
        "break" => TT_BREAK,
        "continue" => TT_CONTINUE,
        "const" => TT_CONST,
        "case" => TT_CASE,
        "char" => TT_CHAR,
        "do" => TT_DO,
        "double" => TT_DOUBLE,
        "default" => TT_DEFAULT,
        "enum" => TT_ENUM,
        "else" => TT_ELSE,
        "extern" => TT_EXTERN,
        "float" => TT_FLOAT,
        "for" => TT_FOR,
        "goto" => TT_GOTO,
        "int" => TT_INT,
        "if" => TT_IF,
        "long" => TT_LONG,
        "return" => TT_RETURN,
        "register" => TT_REGISTER,
        "struct" => TT_STRUCT,
        "signed" => TT_SIGNED,
        "sizeof" => TT_SIZEOF,
        "static" => TT_STATIC,
        "short" => TT_SHORT,
        "switch" => TT_SWITCH,
        "typedef" => TT_TYPEDEF,
        "union" => TT_UNION,
        "unsigned" => TT_UNSIGNED,
        "void" => TT_VOID,
        "volatile" => TT_VOLATILE,
        "while" => TT_WHILE,
        "&&" => TT_LAND,
        "||" => TT_LOR,
        "-=" => TT_DEC,
        "+=" => TT_INC,
        "++" => TT_PLUSPLUS,
        "--" => TT_MINUSMINUS,
        "/=" => TT_DIVEQ,
        "*=" => TT_MULEQ,
        "%=" => TT_MODEQ,
        "&=" => TT_BANDEQ,
        "|=" => TT_BOREQ,
        "&&=" => TT_LANDEQ,
        "||=" => TT_LOREQ,
        "<=" => TT_LESSEQ,
        ">=" => TT_GREATEREQ,
        "<<" => TT_LEFTSHIFT,
        ">>" => TT_RIGHTSHIFT,
        "==" => TT_EQUALS,
        "^=" => TT_XOREQ,
        "->" => TT_POINT,
        "<<=" => TT_LEFTSHIFTEQUALS,
        ">>=" => TT_RIGHTSHIFTEQUALS,
        "!=" => TT_NOTEQ,
        _ => TT_NO_TOKEN,
    };
    if keyword_or_operator != TT_NO_TOKEN {
        return keyword_or_operator;
    }

    let mut all_numeric = true;
    let mut count_us = 0;
    for c in contents.chars() {
        if c == '.' || c == '\'' || c == '"' {
            return TT_LITERAL;
        }
        if c == 'u' {
            count_us += 1;
        }
        if !c.is_ascii_digit() && c != 'u' {
            all_numeric = false;
        }
    }
    if all_numeric && (count_us == 0 || (count_us == 1 && contents.ends_with('u'))) {
        TT_LITERAL
    } else {
        TT_IDENTIFIER
    }
}

/// Derives a TokenType from a string input.
pub fn ttype_from_string(contents: &str) -> TokenType {
    let mut chars = contents.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => ttype_one_char(c),
        _ => ttype_many_chars(contents),
    }
}

/// Returns the name of a token type as a string.
pub fn ttype_name(tt: TokenType) -> String {
    use TokenType::*;
    let name = match tt {
        TT_LITERAL => "literal",
        TT_IDENTIFIER => "identifier",
        TT_OPAREN => "open paren",
        TT_CPAREN => "close paren",
        TT_OBRACE => "open brace",
        TT_CBRACE => "close brace",
        TT_OBRACKET => "open bracket",
        TT_CBRACKET => "close bracket",
        TT_SEMI => "semicolon",
        TT_NO_TOKEN => "no token",
        TT_EOF => "end of file",
        TT_NEWLINE => "newline",
        TT_POUND => "pound",
        TT_PERIOD => ".",
        TT_COMMA => ",",
        TT_QMARK => "?",
        TT_MINUS => "-",
        TT_PLUS => "+",
        TT_STAR => "*",
        TT_SLASH => "/",
        TT_ASSIGN => "=",
        TT_COLON => ":",
        TT_MOD => "%",
        TT_BAND => "&",
        TT_LAND => "&&",
        TT_BOR => "|",
        TT_LOR => "||",
        TT_DEC => "-=",
        TT_INC => "+=",
        TT_PLUSPLUS => "++",
        TT_MINUSMINUS => "--",
        TT_DIVEQ => "/=",
        TT_MULEQ => "*=",
        TT_MODEQ => "%=",
        TT_BANDEQ => "&=",
        TT_BOREQ => "|=",
        TT_LANDEQ => "&&=",
        TT_LOREQ => "||=",
        TT_GREATER => ">",
        TT_LESS => "<",
        TT_LESSEQ => "<=",
        TT_GREATEREQ => ">=",
        TT_LEFTSHIFT => "<<",
        TT_RIGHTSHIFT => ">>",
        TT_LNOT => "!",
        TT_BNOT => "~",
        TT_EQUALS => "==",
        TT_NOTEQ => "!=",
        TT_XOR => "^",
        TT_XOREQ => "^=",
        TT_POINT => "->",
        TT_LEFTSHIFTEQUALS => "<<=",
        TT_RIGHTSHIFTEQUALS => ">>=",
        TT_AUTO => "auto",
        TT_BREAK => "break",
        TT_CHAR => "char",
        TT_CONST => "const",
        TT_CASE => "case",
        TT_CONTINUE => "continue",
        TT_DOUBLE => "double",
        TT_DO => "do",
        TT_DEFAULT => "default",
        TT_ENUM => "enum",
        TT_ELSE => "else",
        TT_EXTERN => "extern",
        TT_FLOAT => "float",
        TT_FOR => "for",
        TT_GOTO => "goto",
        TT_IF => "if",
        TT_INT => "int",
        TT_LONG => "long",
        TT_RETURN => "return",
        TT_REGISTER => "register",
        TT_STATIC => "static",
        TT_SWITCH => "switch",
        TT_SHORT => "short",
        TT_SIGNED => "signed",
        TT_STRUCT => "struct",
        TT_SIZEOF => "sizeof",
        TT_TYPEDEF => "typedef",
        TT_UNSIGNED => "unsigned",
        TT_UNION => "union",
        TT_VOID => "void",
        TT_VOLATILE => "volatile",
        TT_WHILE => "while",
    };
    name.to_string()
}

/// Pushes a token back into the lexer's buffer.
pub fn unlex(l: &mut Lexer, t: &Token) -> i32 {
    if l.unlexed_count as usize >= TOKEN_PUTBACKS {
        eprintln!("\x1b[31mError: jccc: internal: tried to unlex more than {TOKEN_PUTBACKS} tokens at a time\x1b[0m");
        return -1;
    }
    l.unlexed[l.unlexed_count as usize] = t.clone();
    l.unlexed_count += 1;
    0
}

/// Skips characters until the next token is found.
pub fn skip_to_token(l: &mut Lexer) -> i32 {
    let mut current = lexer_getchar(l);
    if current == -1 {
        return -1;
    }
    let mut previous = current;
    if !matches!(current, c if c == b' ' as i32 || c == b'\t' as i32 || c == b'/' as i32) {
        lexer_ungetchar(l);
        return 0;
    }

    let mut in_block = 0;
    let mut pass = 0;
    loop {
        current = lexer_getchar(l);
        if current == -1 {
            return -1;
        }
        let slash = b'/' as i32;
        let star = b'*' as i32;
        if current == slash && previous == slash && in_block == 0 {
            in_block = 1;
        } else if current == star && previous == slash && in_block == 0 {
            in_block = 2;
            pass = 2;
        } else if (in_block == 1 && current == b'\n' as i32)
            || (in_block == 2 && current == slash && previous == star && pass <= 0)
        {
            in_block = 0;
        } else if previous == slash && current != star && current != slash && in_block == 0 {
            lexer_ungetchar(l);
            return 0;
        }
        if current != b' ' as i32
            && current != b'\t' as i32
            && current != slash
            && in_block == 0
        {
            lexer_ungetchar(l);
            return 0;
        }
        pass -= 1;
        previous = current;
    }
}

fn set_token(t: &mut Token, l: &Lexer, contents: String, token_type: TokenType) {
    t.contents = contents;
    t.length = t.contents.len();
    t.token_type = token_type;
    t.source_file = l.current_file.clone();
    t.line = l.line;
    t.column = l.column;
}

/// Retrieves the next token from the lexer.
pub fn real_lex(l: &mut Lexer, t: &mut Token) -> i32 {
    if l.unlexed_count > 0 {
        l.unlexed_count -= 1;
        *t = l.unlexed[l.unlexed_count as usize].clone();
        return 0;
    }

    skip_to_token(l);
    let init = lexer_getchar(l);
    t.contents.clear();
    t.source_file = l.current_file.clone();
    if init == -1 {
        set_token(t, l, "[end of file]".to_string(), TokenType::TT_EOF);
        return 0;
    }
    if init == b' ' as i32 || init == b'\t' as i32 {
        eprintln!("\x1b[31mError: jccc: internal error: did not skip whitespace correctly\x1b[0m");
        return -1;
    }
    if init == b'\n' as i32 {
        set_token(t, l, "[newline]".to_string(), TokenType::TT_NEWLINE);
        return 0;
    }

    let initial = init as u8 as char;
    if in_string(initial, "(){}[];~#,.:?~") != 0 {
        set_token(t, l, initial.to_string(), ttype_one_char(initial));
        return 0;
    }

    if is_valid_numeric_or_id_char(initial) != 0 {
        let starting_line = l.line;
        let starting_col = l.column;
        let mut contents = initial.to_string();
        loop {
            let next = lexer_getchar(l);
            if next == -1 || is_valid_numeric_or_id_char(next as u8 as char) == 0 {
                break;
            }
            if contents.len() >= TOKEN_LENGTH - 1 {
                eprintln!("\x1b[31mError: jccc: identifier too long, over {TOKEN_LENGTH} characters\x1b[0m");
                return -1;
            }
            contents.push(next as u8 as char);
        }
        lexer_ungetchar(l);
        t.contents = contents;
        t.length = t.contents.len();
        t.token_type = ttype_many_chars(&t.contents);
        t.line = starting_line;
        t.column = starting_col;
        return 0;
    }

    if starts_operator(initial) != 0 {
        let mut contents = initial.to_string();
        while valid_operator_sequence(&contents) != 0 {
            let next = lexer_getchar(l);
            contents.push(next as u8 as char);
        }
        lexer_ungetchar(l);
        contents.pop();
        t.contents = contents;
        t.length = t.contents.len() + 1;
        t.token_type = ttype_from_string(&t.contents);
        return 0;
    }

    eprintln!("\x1b[31mError: jccc: lexer unable to identify token starting with: {initial}\x1b[0m");
    0
}

/// Main lex function to tokenize input into a given Token.
pub fn lex(l: &mut Lexer, token: &mut Token) -> i32 {
    loop {
        let status = real_lex(l, token);
        if status != 0 {
            return status;
        }
        if token.token_type != TokenType::TT_NEWLINE {
            return 0;
        }
    }
}

/// Tests the function that identifies token types by name.
pub fn test_ttype_name() -> i32 {
    println!("Running \"test_ttype_name\"");
    assert_eq!(ttype_name(TokenType::TT_LITERAL), "literal");
    assert_eq!(ttype_name(TokenType::TT_PLUS), "+");
    assert_eq!(ttype_name(TokenType::TT_SIZEOF), "sizeof");
    assert_eq!(ttype_name(TokenType::TT_WHILE), "while");
    0
}

/// Tests the function that determines token types for a single character.
pub fn test_ttype_one_char() -> i32 {
    println!("Running \"test_ttype_one_char\"");
    assert_eq!(ttype_one_char('a'), TokenType::TT_IDENTIFIER);
    assert_eq!(ttype_one_char('1'), TokenType::TT_LITERAL);
    assert_eq!(ttype_one_char('+'), TokenType::TT_PLUS);
    assert_eq!(ttype_one_char('-'), TokenType::TT_MINUS);
    assert_eq!(ttype_one_char('>'), TokenType::TT_GREATER);
    assert_eq!(ttype_one_char('~'), TokenType::TT_BNOT);
    0
}

/// Tests the function that determines token types from a string.
pub fn test_ttype_from_string() -> i32 {
    println!("Running \"test_ttype_from_string\"");
    assert_eq!(ttype_from_string("+"), TokenType::TT_PLUS);
    assert_eq!(ttype_from_string("="), TokenType::TT_ASSIGN);
    assert_eq!(ttype_from_string("1"), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("1.2"), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("1u"), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("1.2f"), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("1.f"), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("\"Planck\""), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("'Language'"), TokenType::TT_LITERAL);
    assert_eq!(ttype_from_string("Jaba"), TokenType::TT_IDENTIFIER);
    assert_eq!(ttype_from_string("cat_"), TokenType::TT_IDENTIFIER);
    assert_eq!(ttype_from_string("("), TokenType::TT_OPAREN);
    assert_eq!(ttype_from_string("}"), TokenType::TT_CBRACE);
    assert_eq!(ttype_from_string(";"), TokenType::TT_SEMI);
    0
}

/// Tests the function that determines token types from multiple characters.
pub fn test_ttype_many_chars() -> i32 {
    println!("Running \"test_ttype_many_chars\"");
    assert_eq!(ttype_many_chars("foo"), TokenType::TT_IDENTIFIER);
    assert_eq!(ttype_many_chars("struct"), TokenType::TT_STRUCT);
    assert_eq!(ttype_many_chars("while"), TokenType::TT_WHILE);
    0
}
