#ifndef IMAGE_H
#define IMAGE_H

struct Image
{
    Image();
    Image(int w, int h);
    Image(const Image& rhs);
    ~Image();
    const Image& operator = (const Image& rhs);

    unsigned char* GetPixelAt(int x, int y);

    int width, height;
    unsigned char* buf;
};

//----------------------------------------------------------------------------
Image::Image()
{
    width = 0;
    height = 0;
    buf = NULL;
}

//----------------------------------------------------------------------------
Image::Image(int w, int h)
{
    width = w;
    height = h;
    buf = new unsigned char[3 * w * h];
}

//----------------------------------------------------------------------------
Image::Image(const Image& rhs)
{
    width = rhs.width;
    height = rhs.height;
    
    int totalBufSize = 3 * rhs.width * rhs.height;
    buf = new unsigned char[totalBufSize];

    memcpy(buf, rhs.buf, totalBufSize);
}

//----------------------------------------------------------------------------
Image::~Image()
{
    delete[] buf;
}

//----------------------------------------------------------------------------
// Handles self assignment (implicit) and is exception safe. 
// See this for more details:
// http://www.parashift.com/c++-faq-lite/assignment-operators.html#faq-12.3
//----------------------------------------------------------------------------
const Image& Image::operator = (const Image& rhs)
{
    unsigned char* origBuf = buf;
    
    int totalBufSize = 3 * rhs.width * rhs.height;
    buf = new unsigned char[totalBufSize];

    memcpy(buf, rhs.buf, totalBufSize);

    delete[] origBuf;

    width = rhs.width;
    height = rhs.height;

    return *this;
}

//----------------------------------------------------------------------------
// Get the pointer p to the red color component of pixel at (x,y).
// Note: p[0], p[1], p[2] are r,g,b components of pixel at location (x,y).
//----------------------------------------------------------------------------
unsigned char* Image::GetPixelAt(int x, int y)
{
    // index checking... Remove this in release version for efficiency
    if (x < 0 || x > width - 1 || y < 0 || y > height - 1)
    {
        fprintf(stderr, "x: %d y: %d -> Invalid index\n", x, y);
        exit(1);
    }
    int i = 3 * (x + y * width);
    
    return &buf[i];
}

#endif