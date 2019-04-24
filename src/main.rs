use std::sync;
use std::thread;

fn run_mono() {
    let mut x = vec![1, 2, 3, 6];
    thread::spawn( || {
        x.push(6);
    });
}



fn main() {
    println!("lollipop\n");
}
