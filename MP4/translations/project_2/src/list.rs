use std::any::Any;

/// A block in a linked list that holds multiple elements.
#[derive(Debug)]
pub struct ListBlock {
    pub array: Vec<Box<dyn Any>>,
    pub size: i32,
    pub full: i32,
    pub next: Option<Box<ListBlock>>,
}

/// A linked list structure consisting of blocks.
#[derive(Debug)]
pub struct List {
    pub head: Option<Box<ListBlock>>,
    /// Retained for interface compatibility; safe traversal uses `head` instead.
    pub tail: Option<*mut ListBlock>,
    pub blocksize: i32,
}

fn make_block(blocksize: i32) -> Box<ListBlock> {
    Box::new(ListBlock {
        array: Vec::with_capacity(blocksize.max(0) as usize),
        size: blocksize,
        full: 0,
        next: None,
    })
}

fn last_block_mut(list: &mut List) -> Option<&mut ListBlock> {
    let mut block = list.head.as_deref_mut()?;
    while block.next.is_some() {
        block = block.next.as_deref_mut()?;
    }
    Some(block)
}

fn block_at_mut(list: &mut List, index: i32) -> Option<(&mut ListBlock, usize)> {
    if index < 0 {
        eprintln!("\x1b[31mError: jccc: internal: list index was negative ({index})\x1b[0m");
        return None;
    }

    let mut remaining = index as usize;
    let mut block = list.head.as_deref_mut()?;
    loop {
        let size = block.size.max(0) as usize;
        if remaining < size {
            return Some((block, remaining));
        }
        remaining = remaining.checked_sub(size)?;
        block = block.next.as_deref_mut()?;
    }
}

/// Retrieves an element from the list by index, if it exists.
pub fn lget_element(l: &mut List, index: i32) -> Option<&mut Box<dyn Any>> {
    let (block, offset) = block_at_mut(l, index)?;
    if offset >= block.full.max(0) as usize {
        eprintln!("\x1b[31mError: jccc: internal: list index {index} out of bounds\x1b[0m");
        return None;
    }
    block.array.get_mut(offset)
}

/// Destroys the list and frees resources.
pub fn destroy_list(l: &mut List) -> i32 {
    // Unlink iteratively so even a very long list does not recurse on drop.
    let mut next = l.head.take();
    while let Some(mut block) = next {
        next = block.next.take();
    }
    l.tail = None;
    0
}

/// Adds an element to the list.
pub fn ladd_element(l: &mut List, element: Box<dyn Any>) -> i32 {
    if l.blocksize <= 0 {
        eprintln!("\x1b[31mError: jccc: internal: list blocksize must be positive\x1b[0m");
        return -1;
    }
    if l.head.is_none() {
        l.head = Some(make_block(l.blocksize));
    }

    let needs_block = match last_block_mut(l) {
        Some(block) => block.full >= block.size,
        None => return -1,
    };
    if needs_block {
        let blocksize = l.blocksize;
        if let Some(block) = last_block_mut(l) {
            block.next = Some(make_block(blocksize));
        }
    }
    if let Some(block) = last_block_mut(l) {
        block.array.push(element);
        block.full += 1;
        0
    } else {
        -1
    }
}

/// Allocates a new block and links it into the list.
pub fn new_block(l: &mut List) -> Box<ListBlock> {
    let blocksize = l.blocksize;
    let block = make_block(blocksize);
    if l.head.is_none() {
        l.head = Some(make_block(blocksize));
    } else if let Some(tail) = last_block_mut(l) {
        tail.next = Some(make_block(blocksize));
    }
    block
}

/// Iterates over the list with a provided function.
pub fn literate(l: &mut List, func: fn(&mut Box<dyn Any>) -> i32) -> i32 {
    let mut sum = 0i32;
    let mut block = l.head.as_deref_mut();
    while let Some(current) = block {
        for element in current.array.iter_mut().take(current.full.max(0) as usize) {
            sum = sum.wrapping_add(func(element));
        }
        block = current.next.as_deref_mut();
    }
    sum
}

/// Finds and sets index variables for internal iteration.
pub fn lfind_index(l: &mut List, lb: &mut Option<Box<ListBlock>>, i: &mut i32) -> i32 {
    if *i < 0 {
        eprintln!(
            "\x1b[31mError: jccc: internal: list index was negative ({})\x1b[0m",
            *i
        );
        return -1;
    }

    let mut remaining = *i as usize;
    let mut source = l.head.as_ref();
    while let Some(block) = source {
        let size = block.size.max(0) as usize;
        if remaining < size {
            *i = remaining as i32;
            *lb = Some(Box::new(ListBlock {
                array: Vec::new(),
                size: block.size,
                full: block.full,
                next: None,
            }));
            return 0;
        }
        remaining -= size;
        source = block.next.as_ref();
    }

    *lb = None;
    eprintln!(
        "\x1b[31mError: jccc: internal: list index {} out of bounds\x1b[0m",
        *i
    );
    -1
}

/// Creates a new list with the specified blocksize.
pub fn create_list(blocksize: i32) -> List {
    List {
        head: None,
        tail: None,
        blocksize,
    }
}

/// Sets an element in the list by index.
pub fn lset_element(l: &mut List, index: i32, value: Box<dyn Any>) -> i32 {
    let Some((block, offset)) = block_at_mut(l, index) else {
        return -1;
    };
    if offset >= block.full.max(0) as usize {
        eprintln!("\x1b[31mError: jccc: internal: list index {index} out of bounds\x1b[0m");
        return -1;
    }
    if let Some(slot) = block.array.get_mut(offset) {
        *slot = value;
        0
    } else {
        -1
    }
}
