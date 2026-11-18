use std::any::Any;

/// Represents a node in the bucket chain.
#[derive(Debug)]
pub struct BucketNode {
    pub key: String,
    pub value: Box<dyn Any>,
    pub next: Option<Box<BucketNode>>,
}

/// Represents a simple Hashmap structure.
#[derive(Debug)]
pub struct Hashmap {
    pub buckets: Vec<Option<Box<BucketNode>>>,
    pub size: i32,
    pub cap: i32,
    pub hash: fn(&str) -> u32,
    /// An equality function taking two &str and returning a bool (true if equal).
    pub equals: fn(&str, &str) -> bool,
}

pub fn create_hashmap(cap: i32) -> Hashmap {
    let capacity = cap.max(0);
    Hashmap {
        buckets: Vec::with_capacity(capacity as usize),
        size: 0,
        cap: capacity,
        hash: fnva1,
        equals: equal_key,
    }
    .with_empty_buckets()
}

trait EmptyBuckets {
    fn with_empty_buckets(self) -> Self;
}

impl EmptyBuckets for Hashmap {
    fn with_empty_buckets(mut self) -> Self {
        self.buckets.resize_with(self.cap as usize, || None);
        self
    }
}

fn bucket_index(h: &Hashmap, key: &str) -> Option<usize> {
    if h.cap <= 0 {
        None
    } else {
        Some(((h.hash)(key) % h.cap as u32) as usize)
    }
}

/// Gets a value from the Hashmap.
pub fn hm_get<'a>(h: &'a Hashmap, key: &'a str) -> Option<&'a BucketNode> {
    let index = bucket_index(h, key)?;
    let mut node = h.buckets.get(index)?.as_deref();
    while let Some(current) = node {
        if (h.equals)(&current.key, key) {
            return Some(current);
        }
        node = current.next.as_deref();
    }
    None
}

/// Sets a key-value pair in the Hashmap.
pub fn hm_set(h: &mut Hashmap, key: &str, value: Box<dyn Any>) -> i32 {
    if h.cap <= 0 {
        return -1;
    }
    if h.size >= h.cap {
        double_cap(h);
    }
    let Some(index) = bucket_index(h, key) else {
        return -1;
    };

    let mut link = &mut h.buckets[index];
    loop {
        match link {
            Some(node) if (h.equals)(&node.key, key) => return -1,
            Some(node) => link = &mut node.next,
            None => break,
        }
    }
    *link = Some(Box::new(BucketNode {
        key: key.to_string(),
        value,
        next: None,
    }));
    h.size += 1;
    0
}

/// Tests setting and getting a value in the Hashmap.
pub fn test_hash_set_and_get() -> i32 {
    println!("Running \"test_hash_set_and_get\"");
    let mut h = create_hashmap(100);
    assert_ne!(hm_set(&mut h, "test", Box::new("jake".to_string())), -1);
    let got = hm_get(&h, "test").expect("stored key must be retrievable");
    assert_eq!(got.value.downcast_ref::<String>(), Some(&"jake".to_string()));
    0
}

/// Doubles the capacity of the Hashmap.
pub fn double_cap(h: &mut Hashmap) {
    if h.cap < 0 {
        return;
    }
    let new_cap = h.cap.saturating_mul(2);
    if new_cap == 0 {
        return;
    }
    let old_buckets = std::mem::take(&mut h.buckets);
    h.buckets = Vec::new();
    h.buckets.resize_with(new_cap as usize, || None);
    h.cap = new_cap;

    for bucket in old_buckets.into_iter().flatten() {
        let mut nodes = Some(bucket);
        while let Some(mut node) = nodes {
            nodes = node.next.take();
            let index = ((h.hash)(&node.key) % h.cap as u32) as usize;
            node.next = h.buckets[index].take();
            h.buckets[index] = Some(node);
        }
    }
}

/// A sample FNV-1 hash function.
pub fn fnva1(value: &str) -> u32 {
    let mut hash = 16_777_619u64;
    for byte in value.bytes() {
        // The original uses a signed `char` on its target platform.
        let signed_byte = byte as i8 as i64 as u64;
        hash ^= signed_byte;
        hash = hash.wrapping_mul(2_166_136_261);
    }
    hash as u32
}

/// Tests initializing the Hashmap.
pub fn test_hash_init() -> i32 {
    println!("Running \"test_hash_init\"");
    let h = create_hashmap(100);
    assert_eq!(h.size, 0);
    assert_eq!(h.cap, 100);
    0
}

/// Destroys the Hashmap and frees resources.
pub fn destroy_hashmap(h: &mut Hashmap) {
    h.buckets.clear();
    h.size = 0;
    h.cap = 0;
}

/// Tests initializing the Hashmap and storing a value.
pub fn test_hash_init_and_store() -> i32 {
    println!("Running \"test_hash_init_and_store\"");
    let mut h = create_hashmap(100);
    assert_eq!(h.size, 0);
    assert_eq!(h.cap, 100);
    assert_ne!(hm_set(&mut h, "test", Box::new("jake".to_string())), -1);
    let index = bucket_index(&h, "test").expect("nonzero capacity");
    assert_eq!(h.buckets[index].as_ref().map(|node| node.key.as_str()), Some("test"));
    assert_eq!(h.size, 1);
    assert_eq!(h.cap, 100);
    0
}

/// Tests setting a key-value pair, then doubling capacity, then getting the value.
pub fn test_hash_set_and_double_get() -> i32 {
    println!("Running \"test_hash_set_and_double_get\"");
    let mut h = create_hashmap(100);
    assert_ne!(hm_set(&mut h, "test", Box::new("jake".to_string())), -1);
    double_cap(&mut h);
    let got = hm_get(&h, "test").expect("stored key must survive rehashing");
    assert_eq!(got.value.downcast_ref::<String>(), Some(&"jake".to_string()));
    0
}

/// Compares two string keys for equality.
pub fn equal_key(a: &str, b: &str) -> bool {
    a == b
}
