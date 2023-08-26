usage
=====

```rust
use tiny_png::{read_png, write_apng, Image, ImageData};

fn main() {
    // load an image
    let image = read_png(&"test.png".to_string()).expect("can't load test.png");

    println!("{} x {}", image.width(), image.height());

    let data = image.data();

    (0 .. image.height()).for_each(|y| {
      (0 .. image.width()).for_each(|x| {
        let pixel = data[y][x]; // (u16, u16, u16, u16)
      });
    });

    // now write it back as one-frame image
  
    write_apng(&"back.png".to_string(), 
        &ImageData::RGBA16(vec![data]),
        None,
        None,
        false).expect("can't save back.png");
}
```

supported formats 
=================

| ImageData::RGB | 8-bit RGB without alpha |
| ImageData::RGBA | 8-bit RGB with alpha |
| ImageData::RGB16 | 16-bit plain RGB without alpha |
| ImageData::RGBA16 | 16-bit RGB with alpha |
| ImageData::NDX | 8-bit indexed palette without alpha |
| ImageData::NDXA | 8-bit indexed palette with alpha |

