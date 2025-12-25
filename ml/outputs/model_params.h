#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

/*
 * Auto-generated model parameters for gesture recognition
 * Model: LogisticRegression with 18-dim features (mean/std/energy per channel)
 * Labels: ['circle', 'shake', 'still']
 * Window: 0.5s (26 samples @ ~52.0Hz)
 * Hop: 0.10000000149011612s (5 samples)
 */

#define NUM_CLASSES 3
#define NUM_FEATURES 18
#define NUM_CHANNELS 6
#define WIN_N 26
#define HOP_N 5

// Label names
static const char* LABEL_NAMES[NUM_CLASSES] = {"circle", "shake", "still"};

// Normalization parameters (per channel, for raw window data)
static const float NORM_MEAN[6] = {1.191957, -1.500641, 9.366179, 0.005571, 0.006305, -0.001824}; // shape: (6,)
static const float NORM_STD[6] = {3.576995, 3.629982, 7.986805, 2.187417, 1.051818, 1.119035}; // shape: (6,)

// LogisticRegression weights
static const float LR_COEF[54] = {-0.542950, -0.132968, 0.215510, -0.134080, 0.220683, -0.506905, 1.485399, 0.649368, -0.281935, 0.021528, 0.813800, 1.951077, 0.717341, -0.779632, -0.825704, -0.562254, -0.102496, 1.667681, 1.551375, 0.268938, -0.510996, 0.412295, 0.167280, 0.075157, 0.239605, 0.429442, 1.488263, 1.134873, 1.195834, -0.100847, 0.333416, 1.657031, 1.174793, 0.965213, 1.157585, -0.541559, -1.008425, -0.135970, 0.295486, -0.278214, -0.387964, 0.431748, -1.725004, -1.078811, -1.206328, -1.156402, -2.009634, -1.850230, -1.050756, -0.877399, -0.349089, -0.402959, -1.055089, -1.126122}; // shape: (3, 18)  // [NUM_CLASSES, NUM_FEATURES]
static const float LR_INTERCEPT[3] = {-1.800370, -2.511089, 4.311459}; // shape: (3,)

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
