#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "matrix.h"
#include "argvparser.h"

#define NUM 7 
#define GIST 16 
#define PI 3.141592653589793
#define COL 8
#define ASTEP 0.4
#define EPS 0.0001

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tie;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;
typedef Matrix<float> Image;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

Image CreateGreyscale(pair<BMP *, int> data)
{
	RGBApixel pixel;
	BMP *bmp_im;
	int label;
	tie(bmp_im, label) = data;
	Image im(bmp_im->TellWidth(), bmp_im->TellHeight());
	for(uint i = 0; i < im.n_rows; ++i)
		for(uint j = 0; j < im.n_cols; ++j){
			pixel = bmp_im -> GetPixel(i, j);
			im(i, j) = 0.299*pixel.Red + 0.587*pixel.Blue + 0.114*pixel.Green;
		}
	return im.deep_copy();
}

pair<Image, Image> MakeSobel(const Image& im)
{
	Image hor_sobel(im.n_rows - 2, im.n_cols - 2);
	Image ver_sobel(im.n_rows - 2, im.n_cols - 2);
	Image sobel_x = {	{-1, 0, 1},
						{-2, 0, 2},
						{-1, 0, 1}};
	Image sobel_y = {	{-1, -2, -1},
						{0, 0, 0},
						{1, 2, 1}};
	for(uint i = 0; i < hor_sobel.n_rows; ++i)
		for(uint j = 0; j < hor_sobel.n_cols; ++j){
			hor_sobel(i, j) = sobel_x * im.submatrix(i, j, 3, 3);
			ver_sobel(i, j) = sobel_y * im.submatrix(i, j, 3, 3);
		}
	return make_pair(hor_sobel.deep_copy(), ver_sobel.deep_copy());
}

pair<Image, Image> SolveGrad(const Image& hor, const Image& ver)
{
	Image modul(hor.n_rows, hor.n_cols);
	Image direct(hor.n_rows, hor.n_cols);
	for(uint i = 0; i < modul.n_rows; ++i)
		for(uint j = 0; j < modul.n_cols; ++j){
			modul(i, j) = sqrt(hor(i,j)*hor(i,j) + ver(i,j)*ver(i,j));
			direct(i, j) = atan2(ver(i,j), hor(i,j));
		}
	return make_pair(modul.deep_copy(), direct.deep_copy());
}

vector<float> MakeNullVector()
{
	vector<float> gist;
	for(int i = 0; i < GIST; ++i)
		gist.push_back(0.0);
	return gist;
}

vector<float> Normilize(vector<float> gist)
{
	int norm = 0;
	for(int i = 0; i < GIST; ++i)
		norm += gist[i] * gist[i];
	norm = sqrt(norm);
	if(norm != 0)
		for(int i = 0; i < GIST; ++i){
			gist[i] /= norm;
		}
	return gist;
}

vector<float> GistToSq(const Image& modul, const Image& direct)
{
	vector<float> gist32 = MakeNullVector();
	for(uint i = 0; i < modul.n_rows; ++i)
		for(uint j = 0; j < direct.n_cols; ++j){
			gist32[(direct(i, j) / PI + 1)*(GIST / 2 - 0.5)] += modul(i, j);
		}
	gist32 = Normilize(gist32);
	return gist32;
}

vector<float> ConcatGist(const vector<float> gist, vector<float> all_gist)
{
	for(uint i = 0; i < gist.size(); ++i){ 
		all_gist.push_back(gist[i]);	
	}
	return all_gist;
}

vector<float> MakeGist(const Image& modul, const Image& direct)
{
	vector<float> gist_sq, all_gist;
	uint n_pix_rows, n_pix_cols, num_r, num_c, beg_i, beg_j;
	n_pix_rows = modul.n_rows / NUM;
	num_r = NUM;
	n_pix_cols = modul.n_cols / NUM;
	num_c = NUM;
	int rows_ost;
	int cols_ost;
	for(uint i = 0; i < num_r; ++i)
		for(uint j = 0; j < num_c; ++j){
			beg_i = i*n_pix_rows;
			beg_j = j*n_pix_cols;
			if(i == num_r - 1)
				rows_ost = modul.n_rows % NUM;
			else
				rows_ost = 0;
			if(i == num_c - 1)
				cols_ost = modul.n_cols % NUM;
			else
				cols_ost = 0;
			gist_sq = GistToSq(
				modul.submatrix(beg_i, beg_j, n_pix_rows + rows_ost, n_pix_cols + cols_ost),
				direct.submatrix(beg_i, beg_j, n_pix_rows + rows_ost, n_pix_cols + cols_ost));
			all_gist = ConcatGist(gist_sq, all_gist);
			gist_sq.clear();
		}
	return all_gist;
}

vector<float> MakeGistPiramid(const Image& modul, const Image& direct)
{
	vector<float> gist_0, gist_1, gist_2, gist_3, gist_4, gist_all;
	uint rows = modul.n_rows, cols = modul.n_cols;
	uint q1_rows = rows / 2, q1_cols = cols / 2;
	uint q2_rows = q1_rows + rows % 2, q2_cols = q1_cols + cols % 2;
	gist_0 = MakeGist(modul, direct);
	gist_1 = MakeGist(modul.submatrix(0, 0, q1_rows, q1_cols),
				direct.submatrix(0, 0, q1_rows, q1_cols));
	gist_2 = MakeGist(modul.submatrix(0, q1_cols, q1_rows, q2_cols),
					direct.submatrix(0, q1_cols, q1_rows, q2_cols));
	gist_3 = MakeGist(modul.submatrix(q1_rows, 0, q2_rows, q1_cols),
					direct.submatrix(q1_rows, 0, q2_rows, q1_cols));
	gist_4 = MakeGist(modul.submatrix(q1_rows, q1_cols, q2_rows, q2_cols),
					direct.submatrix(q1_rows, q1_cols, q2_rows, q2_cols));
	gist_all = ConcatGist(gist_0, gist_all);
	gist_all = ConcatGist(gist_1, gist_all);
	gist_all = ConcatGist(gist_2, gist_all);
	gist_all = ConcatGist(gist_3, gist_all);
	gist_all = ConcatGist(gist_4, gist_all);
	return gist_all;
}

vector<float> GetColor(int i, int j, BMP *bm, vector<float>& color)
{
	RGBApixel pixel;
	pixel = bm -> GetPixel(i, j);
	color[0] += pixel.Red;
	color[1] += pixel.Green;
	color[2] += pixel.Blue;
	return color;
}

vector<float> FindColor(BMP *bm, int step_x, int step_y, int i, int j)
{
	vector<float> colors(3);
	colors[0] = colors[1] = colors[2] = 0;
	int n = 0, rows_ost, cols_ost;
	if(i == COL - 1)
		rows_ost = (bm->TellWidth()) % COL;
	else
		rows_ost = 0;
	if(j == COL - 1)
		cols_ost = (bm->TellHeight()) % COL;
	else
		cols_ost = 0;
	for(int i1 = i*step_x; i1 < (i+1)*step_x + rows_ost; ++i1)
		for(int j1 = j*step_y; j1 < (j+1)*step_y + cols_ost; ++j1){
			colors = GetColor(i1, j1, bm, colors);
			++n;
		}
	for(int i1 = 0; i1 < 3; i1++)
		colors[i1] /= n*255;
	return colors;
}

vector<float> MakeColorGist(BMP *bm)
{
	vector<float> color_vec;
	uint width = bm -> TellWidth();
	uint height = bm -> TellHeight();
	int step_x = width / COL;
	int step_y = height / COL;
	for(int i = 0; i < COL; ++i)
		for(int j = 0; j < COL; ++j){
			color_vec = ConcatGist(FindColor(bm, step_x, step_y, i, j), color_vec);
		}
	return color_vec;
}

float K(float lam)
{
	return 1/cosh(PI*lam);
}

vector<float> MakeSVM(vector<float> gist)
{
	vector<float> im_gist, re_gist, gist_all;
	float x;
	re_gist.clear();
	for(uint i = 0; i < gist.size(); ++i){
		if(gist[i] < 0)
			x = -gist[i];
		else
			x = gist[i];
		if(x < EPS){
			re_gist.push_back(0.0);
			re_gist.push_back(0.0);
			re_gist.push_back(0.0);
			re_gist.push_back(0.0);
			re_gist.push_back(0.0);
			im_gist.push_back(0.0);
			im_gist.push_back(0.0);
			im_gist.push_back(0.0);
			im_gist.push_back(0.0);
			im_gist.push_back(0.0);
		}else{
			re_gist.push_back(cos(2*ASTEP*log(x))*sqrt(x*K(-ASTEP)));	
			im_gist.push_back(sin(2*ASTEP*log(x))*sqrt(x*K(-ASTEP)));	
			re_gist.push_back(cos(ASTEP*log(x))*sqrt(x*K(-ASTEP)));	
			im_gist.push_back(sin(ASTEP*log(x))*sqrt(x*K(-ASTEP)));	
			re_gist.push_back(sqrt(x*K(0.0)));
			im_gist.push_back(0.0);
			re_gist.push_back(cos(-ASTEP*log(x))*sqrt(x*K(ASTEP)));
			im_gist.push_back(sin(-ASTEP*log(x))*sqrt(x*K(ASTEP)));
			re_gist.push_back(cos(-2*ASTEP*log(x))*sqrt(x*K(ASTEP)));
			im_gist.push_back(sin(-2*ASTEP*log(x))*sqrt(x*K(ASTEP)));
		}
	}
	gist_all = ConcatGist(re_gist, im_gist);
	return gist_all;
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
	Image im, hor_sobel, ver_sobel, grad_modul, grad_direct;
	vector<float> gistogramm, color_gist;
	int label;
	BMP *bm;
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
		tie(bm, label) = data_set[image_idx];
		im = CreateGreyscale(data_set[image_idx]);
		tie(hor_sobel, ver_sobel) = MakeSobel(im);
		tie(grad_modul, grad_direct) = SolveGrad(hor_sobel, ver_sobel);
		gistogramm = MakeGistPiramid(grad_modul, grad_direct);
		color_gist = MakeColorGist(bm);
		gistogramm = ConcatGist(color_gist, gistogramm);
		gistogramm = MakeSVM(gistogramm);
		features -> push_back(make_pair(gistogramm, label));
		gistogramm.clear();
		color_gist.clear();
        // End of sample code
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2015.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
