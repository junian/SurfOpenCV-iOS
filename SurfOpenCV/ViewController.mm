//
//  ViewController.m
//  SurfOpenCV
//
//  Created by Junian on 3/6/16.
//  Copyright © 2016 Junian. All rights reserved.
//

#import "ViewController.h"
#import <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

@interface ViewController ()
@property (strong, nonatomic) UIImage *image;
@end

@implementation ViewController

double const surfHessianThresh = 300;
bool const surfExtendedFlag = true;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    self.imageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.imageView setBackgroundColor:[UIColor blackColor]];
    self.image = [UIImage imageNamed:@"Lenna.png"];
    [self.imageView setImage:self.image];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (cv::Mat)cvMatFromUIImage:(UIImage*)image{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,     // Pointer to data
                                                    cols,           // Width of bitmap
                                                    rows,           // Height of bitmap
                                                    8,              // Bits per component
                                                    cvMat.step[0],  // Bytes per row
                                                    colorSpace,     // Color space
                                                    kCGImageAlphaNoneSkipLast
                                                    | kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    return cvMat;
}

- (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    
    CGColorSpaceRef colorspace;
    
    if (cvMat.elemSize() == 1) {
        colorspace = CGColorSpaceCreateDeviceGray();
    }else{
        colorspace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Create CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols, cvMat.rows, 8, 8 * cvMat.elemSize(), cvMat.step[0], colorspace, kCGImageAlphaNone | kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    
    // get uiimage from cgimage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorspace);
    return finalImage;
}

- (cv::Mat)ComputeSingleDescriptorsWithMat:(cv::Mat)image{
    
    cv::Mat descs;
    std::vector<cv::KeyPoint> keypoints;

    //initialize Surf detector
    cv::SurfFeatureDetector detector( surfHessianThresh, surfExtendedFlag );
    
    //detect keypoints
    detector.detect( image, keypoints );
    
    //compute descriptor
    detector.compute(image, keypoints, descs);
    
    return descs;
}

- (cv::Mat)ComputeSingleDescriptors:(NSString*)imagePath{
    //Load image from bundle
    UIImage *img = [UIImage imageNamed:imagePath];
    
    //Convert iOS image into OpenCV image (Mat)
    cv::Mat matImg = [self cvMatFromUIImage:img];
    
    //Convert to grayscale
    cv::Mat grayMat;
    cv::cvtColor(matImg, grayMat, CV_BGR2GRAY);
    
    return [self ComputeSingleDescriptorsWithMat:grayMat];
}

- (IBAction)TestClicked:(id)sender {
    cv::Mat originalMat = [self cvMatFromUIImage:self.image];
    
    cv::Mat grayMat;
    cv::cvtColor(originalMat, grayMat, CV_BGR2GRAY);
    
    cv::Mat result = [self ComputeSingleDescriptorsWithMat:grayMat];
    
    // convert gray mat back to UIImage
    [self.imageView setImage:[self UIImageFromCVMat:result]];
}
@end
