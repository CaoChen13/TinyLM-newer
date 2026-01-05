#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

/*
 * Auto-generated model parameters for gesture recognition
 * Model: LogisticRegression with 18-dim features (mean/std/energy per channel)
 * Labels: ['still', 'tilt', 'shake', 'circle', 'tap', 'flip']
 * Window: 0.5s (25 samples @ 50Hz)
 * Hop: 0.1s (5 samples)
 */

#define NUM_CLASSES 6
#define NUM_FEATURES 18
#define NUM_CHANNELS 6
#define WIN_N 25
#define HOP_N 5

// Label names
static const char* LABEL_NAMES[NUM_CLASSES] = {"still", "tilt", "shake", "circle", "tap", "flip"};

// Normalization parameters (per channel, for raw window data)
static const float NORM_MEAN[6] = {-0.662478, 0.108072, 4.623691, 0.061363, -0.017637, 0.036020}; // shape: (6,)
static const float NORM_STD[6] = {7.174473, 6.262358, 6.962134, 1.414809, 1.709603, 1.014076}; // shape: (6,)

// LogisticRegression weights
static const float LR_COEF[108] = {0.862422, -0.135374, 1.041071, 0.215663, 0.416552, -0.651703, -1.763866, -1.775160, -0.889501, -1.409601, -2.012981, -2.858775, -1.597311, -1.143386, 0.285392, -0.301891, -0.686020, -1.564215, -0.024348, 0.194986, -4.880047, 0.234165, 0.276866, 0.939646, -1.765537, -0.338638, 0.585307, -0.931866, -1.471522, -0.847566, 1.449239, 1.902687, -1.892488, 0.427773, 0.525466, -0.121819, -0.549962, -1.007249, -0.556748, -0.199224, -0.206074, -0.598776, -0.514214, -1.877733, -1.365600, 1.126908, 1.857102, 0.780270, 2.153741, -0.266758, -0.704395, 1.937790, 0.599816, 1.064524, -1.478475, 0.094654, 1.364961, 0.622877, 0.410938, -0.220703, 2.720048, 3.505542, -1.968960, 0.828570, -0.803924, 2.519822, 0.048742, 1.635961, -0.826810, -1.700084, -1.568413, 0.588575, -0.312151, -0.649502, 3.121243, -0.600294, -0.409645, 0.781557, 1.642787, 1.224140, 2.467865, -0.487028, 1.266903, -1.224783, -0.332564, 0.290271, 2.127865, -1.337050, -0.104268, -1.702200, 1.502513, 1.502485, -0.090479, -0.273186, -0.488637, -0.250021, -0.319218, -0.738152, 1.170889, 0.873018, 1.164421, 1.631032, -1.721846, -2.418775, 1.010436, 0.973462, 1.233419, 1.735135}; // shape: (6, 18)
static const float LR_INTERCEPT[6] = {1.858656, 4.621846, -1.283091, -0.852464, -2.734994, -1.609953}; // shape: (6,)

/*
 * Feature extraction (18-dim):
 *   For each of 6 channels: mean, std, energy (mean of x^2)
 *   features[0:6]   = mean of each channel
 *   features[6:12]  = std of each channel  
 *   features[12:18] = energy of each channel
 *
 * Prediction:
 *   logits[k] = sum(features[i] * LR_COEF[k*18 + i]) + LR_INTERCEPT[k]
 *   pred = argmax(logits)
 */

#endif // MODEL_PARAMS_H
