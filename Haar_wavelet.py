import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from pywt import dwt2, idwt2
import sys
from skimage.exposure import rescale_intensity
import pickle as pkl

gray_conversion = lambda rgb : np.dot(rgb[...,:3],[0.299 , 0.587, 0.114])


#matrix for conversion of rgb  to yuv
yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])


rgb_from_yuv = np.linalg.inv(yuv_from_rgb)

def psnr(I1,I2):
    if(len(I1.shape)==2 and len(I2.shape)==2):
        MSE = np.mean(np.power((I1-I2),2),dtype=np.float64)
        if(np.max(I1) > 1):
            R=255
        else:
            R=1
        psnr_val =10*np.log10(R**2/MSE)
    return psnr_val


#run length encoding which writes the image into an encoded file
def run_length_encoding(image):
    m,n = image.shape
    image= image.flatten()
    bitstream = ""
    skipped_zeros = 0
    for i in range(image.shape[0]):
        if(image[i]!=0):
            bitstream = bitstream + str(image[i]) + " "+ str(skipped_zeros)+ " "
            skipped_zeros=0
        else:
            skipped_zeros=skipped_zeros+1
    bitstream = str(m)+ " " + str(n) + " "+ bitstream + ";"
    return bitstream



#add gausian noise
def gaussian_noise(img,var=0.001,mean=0):
    image = img.copy()
    if len(image.shape)==2:
        m,n=image.shape
        sigma = var**0.5
        gaussian_matrix = np.random.normal(mean,sigma,(m,n))
        noise_image = image+gaussian_matrix
    if len(image.shape)>=3:
        noise_image = np.copy(image)
        sigma = var**0.5
        gaussian_matrix = np.random.normal(mean,sigma,(image.shape[0],image.shape[1]))
        for i in range(image.shape[2]):
            noise_image[:,:,i]=image[:,:,i]+gaussian_matrix
    return noise_image,gaussian_matrix



'''
 returns discrete haar transformation of an input image for one level in following way
        | LL | LH |
        |____|____|
        | HL | HH |
        |____|____|
'''
def haar2D(image):
    img = image.copy()
    assert len(img.shape)==2
    rows = img.shape[0]
    col= img.shape[1]
    res = img.copy()

    for i in range(rows):
        l = 0
        m = col//2
        for j in range(col//2):
            res[i][l] = (img[i,2*j] + img[i,2*j+1])/np.sqrt(2)
            l= l+1
            res[i][m] = (img[i,2*j]-img[i,2*j+1])/np.sqrt(2)
            m=m+1

    result_img = np.zeros(res.shape)
    for i in range(col):
        for j in range(rows//2):
            result_img[j][i] = (res[2*j][i] +res[2*j+1][i])/np.sqrt(2) 
            result_img[rows//2+j][i] = (res[2*j][i] - res[2*j+1][i])/np.sqrt(2)

    #approximation
    LL = result_img[:rows//2,:col//2]

    #details coef- Horizontal
    LH= result_img[:rows//2,col//2:]

    #details coef - Vertical
    HL = result_img[rows//2:,:col//2]

    #details coef - Diagonal
    HH= result_img[rows//2:,col//2:]
            
    coefficients = dict()
    coefficients["LH"] = LH
    coefficients["HL"] = HL
    coefficients["HH"] = HH
    return result_img,LL,coefficients



'''
perform haar transform of an image for given number of levels
@returns 
    haar_result - a 2d matrix of details and approximation in top left corner
    LL          -  aprromixatio coefficient after haar transformation for given number of levels
    details coefficients
'''
def haar_transform(image,levels=None,K=None): 
    img = image.copy()
    m,n = img.shape
    detail_coef = dict()
    if(levels is not None):
        assert np.power(2,levels)<=m
    haar_result,LL,_coef  = haar2D(img)
    m=m//2
    n = n//2
    level =1
    detail_coef["level_"+str(level)] = _coef
    if(m==n):
        if(levels is not None):     
            while(level!=levels):
                res,L,_coef = haar2D(LL)
                haar_result[:m,:n] = res
                LL = L
                m = m//2
                n = n//2
                level = level+1
                # if(threshold is not None):
                #     for key,value in _coef.items():
                #         _coef[key] = np.where(np.abs(_coef[key])<threshold,0,_coef[key])
                detail_coef["level_"+str(level)] = _coef
        else:
            while(m!=1):
                res,L,_coef = haar2D(LL)
                haar_result[:m,:n] = res
                LL = L
                m = m//2
                n = n//2
                level = level+1
                # if(threshold is not None):
                #     for key,value in _coef.items():
                #         _coef[key] = np.where(np.abs(_coef[key])<threshold,0,_coef[key])
                detail_coef["level_"+str(level)] = _coef
    return haar_result,LL,detail_coef


'''
 @inputs
    a - approximation coefficient matrix
    detail coefficient after haar transformation
    haar_forward - a 2d matrix containing approximation and details coefficients
    K- input for thresholding(retain top K% details coefficients and others are converted to zero)
    softT --> for soft thresholding
    HardT --> for hard thresholding
@returns - details coefficients after thresholding and haar 2d matrix after threshodling
'''
def thresholdingWaveletCoef(a,detail_coef,haar_forward,K):
    total_levels = len(detail_coef)
    k = total_levels
    x,y = a.shape
    haar_forward = haar_forward.astype(np.float32)
    wavelet_coef = np.array(list(haar_forward[x:,y:].flatten()) + list(haar_forward[x:,:y].flatten()))
    wavelet_coef = np.unique(np.round(wavelet_coef,decimals=1))
    threshold_val = np.percentile(np.abs(wavelet_coef),100-K)
    print(threshold_val)
    while(k!=1):
        for key,value in detail_coef["level_"+str(k)].items():
             detail_coef["level_"+str(k)][key] = np.where(np.abs(detail_coef["level_"+str(k)][key])<threshold_val,0,detail_coef["level_"+str(k)][key])
        k = k-1
    haar_forward = np.where(np.abs(haar_forward)<threshold_val,0,haar_forward)
    return detail_coef,haar_forward.astype(np.float32)
    



'''
perform inverse haar 2D 
'''
def inverseHaar2D(a,detail_coef,level=None,threshold=None,softT=False,HardT = False):
    total_levels = len(detail_coef)
    k = total_levels
    for i in range(total_levels):
        detail_coef_level = detail_coef["level_"+str(k)]
        LL = a
  
        LH = detail_coef_level['LH']
        HL = detail_coef_level['HL']
        HH = detail_coef_level['HH']
        if(threshold is not None):
            if(HardT==True):
                LH = np.where(np.abs(LH)>threshold,LH,0)
                HL = np.where(np.abs(HL)>threshold,HL,0)
                HH = np.where(np.abs(HH)>threshold,HH,0)
            if(softT==True):
                LH = np.where(LH>threshold,LH-threshold,LH)
                LH = np.where(np.abs(LH)<threshold,0,LH)
                LH = np.where(LH<(-1*threshold),LH+threshold,LH)

                HL = np.where(HL>threshold,HL-threshold,HL)
                HL = np.where(np.abs(HL)<threshold,0,HL)
                HL = np.where(HL<(-1*threshold),HL+threshold,HL)
            
                HH = np.where(HH>threshold,HH-threshold,HH)
                HH = np.where(np.abs(HH)<threshold,0,HH)
                HH = np.where(HH<(-1*threshold),HH+threshold,HH)
            
        L = np.hstack((LL,LH))
        H = np.hstack((HL,HH))
        haar_t = np.vstack((L,H))

        res = np.zeros(haar_t.shape)
        for i in range(res.shape[1]):
            for j in range(res.shape[0]//2):
                res[2*j][i] = (haar_t[j][i]+haar_t[res.shape[0]//2+j][i])/np.sqrt(2)
                res[2*j+1][i] = (haar_t[j][i]-haar_t[res.shape[0]//2+j][i])/np.sqrt(2)

        new_approx = np.zeros((res.shape[0],res.shape[1])) 
        for i in range(res.shape[0]):
            for j in range(res.shape[1]//2):
                new_approx[i][2*j] = (res[i][j]+res[i][(res.shape[1]//2)+j])/np.sqrt(2)
                new_approx[i][2*j+1] = (res[i][j]-res[i][(res.shape[1]//2)+j])/np.sqrt(2)
        a = new_approx
        k = k-1

    return new_approx

def yuv2rgb(yuv):
    return np.clip(np.dot(yuv,rgb_from_yuv),0,1)



'''
uncompress encoded file(run length encoded file)
'''
def uncompress(file):
    with open(file, "rb") as f:
        stream = pkl.load(f)
        f.close()
    stream = stream.split(" ")
    m = int(stream[0])
    n = int(stream[1])
    img = np.zeros((m*n,))
    i = 2
    k = 0
    skipped_zeros = 0
    while(k<m*n):
        if(stream[i]==";"):
            break
    
        try:
            
            img[k] =float(stream[i])
        except:
            pass

        if(i+3<len(stream)):
            skipped_zeros = int(''.join(filter(str.isdigit, stream[i+3])))
        
        if(skipped_zeros!=0):
            k=k+skipped_zeros+1
        else:
            k=k+1
        i=i+2
    img = img.reshape((m,n))
    return img


'''
@input - detail coefficients and haar_transform 2D matrix
 returns a dictionary of detail coefficients where keys are level number and value are detail coefficient matrix
'''
def haarmatrix2detailCoef(a,haar_transform):
    m,n = haar_transform.shape
    M = m
    N = n
    x,y = a.shape
    detail_coef = dict()
    level=1
    while(m!=x):
        curr_dict = {}
        curr_dict["LH"] = haar_transform[:int(m//2),int(n//2):n]
        curr_dict["HL"] = haar_transform[int(m//2):m,:int(n//2)]
        curr_dict["HH"] = haar_transform[int(m//2):m,int(n//2):n]
        detail_coef["level_"+str(level)] = curr_dict
        level =level+1
        m=int(m//2)
        n = int(n//2)
    return detail_coef


def bitstring_to_bytes(s):
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')

if __name__ == "__main__":

    #read input and converts it into YUV and take only Y-CHANNEL for further processing
    img_file = sys.argv[1]
    orig_img  = plt.imread(img_file)

    img = orig_img.copy()
    if(len(orig_img.shape)>=3):
        orig_img = orig_img[:,:,:3]
        YUV = np.dot(orig_img,yuv_from_rgb)
        orig_img = YUV[:,:,0]

    # cv2.imshow("original Y",orig_img)
    cv2.waitKey(0)

    #add gaussian noise
    orig_noised_img,_ = gaussian_noise(orig_img,var=0.01)
    YUV[:,:,0]=orig_noised_img
    rgb_noised = yuv2rgb(YUV)
    rgb_noised = rescale_intensity(rgb_noised,in_range =(rgb_noised.min(),rgb_noised.max()),out_range=(0,255)).astype('uint8')
    rgb_noised_BGR = cv2.cvtColor(rgb_noised, cv2.COLOR_RGB2BGR)
    # cv2.imshow("original noised",rgb_noised_BGR)
    cv2.waitKey(0)


    #forward wavelet transform
    haar_result,a,detail_coef = haar_transform(orig_noised_img)

    # cv2.imshow("haar_results for noised image",haar_result)
    cv2.waitKey(0)

    #run length encoding
    print("thresholding...")

    '''
    #perform thresholding for details coefficients as given K-value 
    # and return details coefficients in dictionary format for each level and haar_2d matrix
    '''
    detail_coef_after_thresholding,haar_result_after_thresholding = thresholdingWaveletCoef(a,detail_coef,haar_result,20) 
    print("run length encoding...")
    bitstream = run_length_encoding(haar_result_after_thresholding)

    #writing encoded stream to file
    with open("encoded_file.pkl", "wb") as f:
        # f.write(bitstring_to_bytes(bitstream))
        # f.write(bitstr)
        pkl.dump(bitstream, f)
        f.close()

    #decompress image from encoded file as Haar_2D matrix
    rturn_img = uncompress("encoded_file.pkl")

    #get details coefficients from decompress haar_2d matrix and store it into dictionary
    detail_coef = haarmatrix2detailCoef(a,rturn_img)

    #perform inverse haar transform for decompress coefficients
    inv_img = inverseHaar2D(a,detail_coef,HardT=True)
    cv2.imshow('inv_img',inv_img)
    cv2.waitKey(0)
    print("psnr_val",psnr(inv_img,orig_img))
    #perform inverse haar transform for thresholded coefficients
    # inv_img_orig = inverseHaar2D(a,detail_coef_after_thresholding)

    #decompressed image
    YUV[:,:,0] = inv_img
    back_rgb = yuv2rgb(YUV)

    #to check with decompressed, we also perform inverse haar on stored thresholded coefficients
    # YUV[:,:,0] = inv_img_orig
    # back_rgb_orig = yuv2rgb(YUV)
    # plt.imsave(img_file[:-4]+"_inverse_haar_uncompressed.png",back_rgb)
    # plt.imsave(img_file[:-4]+"_inverse_haar_orig.png",back_rgb_orig)
    plt.imsave("uncompressed_result_baboon.png",back_rgb)
    back_rgb =  rescale_intensity(back_rgb,in_range =(back_rgb.min(),back_rgb.max()),out_range=(0,255)).astype('uint8')
    back_bgr = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow("haar",back_bgr)
    cv2.waitKey(0)