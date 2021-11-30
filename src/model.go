package main

import (
	"fmt"

	"os"

	"strconv"

	"encoding/csv"

	"github.com/sjwhitworth/golearn/ensemble"

	"github.com/disintegration/imaging"

	"github.com/sjwhitworth/golearn/base" 

	"image"
)

func main() {
	// Load the trained model
	fmt.Print("Loading model... ")
	model := ensemble.NewRandomForest(32, 48*48)
	model.Load("random_forest_32_trees")

	fmt.Println("Done.")

	// Test various images from google

	img := "image_test/happy.jpg"
	predict_emotion(model, img, "happy")

	img = "image_test/angry.jpg"
	predict_emotion(model, img, "angry")

	img = "image_test/sad.jpg"
	predict_emotion(model, img, "sad")

	img = "image_test/surprise.jpg"
	predict_emotion(model, img, "surprise")

	img = "image_test/surprise2.png"
	predict_emotion(model, img, "surprise")

}


// Given the ML model and a path to an image, predict the emotion in the image
func predict_emotion(model *ensemble.RandomForest, img string, emotion string) {
	// apply image processing and convert the image to an array of pixels
	image_array := imageToArray(img)

	// Save the converted image to a csv
	emotion_csv := "test_image.csv"
	saveImageArrayToCsv(image_array, emotion_csv)
	

	fmt.Print("Loading test image...")

	test_image, error := base.ParseCSVToInstances(emotion_csv, true)
	if error != nil {
		fmt.Println("Error parsing test image csv")
		panic(error)
	}

	fmt.Println("Done.")

	fmt.Println("Predicting emotion...")

	prediction, err := model.Predict(test_image)

	// // make sure the prediction was created without error
	if err != nil {
		fmt.Println("Error predicting emotion")
		panic(err)
	}

	label, err := strconv.Atoi(prediction.RowString(0))

	if err != nil {
		fmt.Println("Error converting label to emotion")
		panic(err)
	}
	
	// print results
	fmt.Println("Prediction: " + get_emotion_from_label(label) + ". Actual: " + emotion)
	fmt.Println()
}


// Saves an image in the form of an array of pixels to a csv
func saveImageArrayToCsv(array []int, file_name string) {
	var headers []string
	var row []string

	for i := 0; i < len(array); i++ {
		column := "pixel" + strconv.Itoa(i)
		headers = append(headers, column)

		row = append(row, strconv.Itoa(array[i]))
	}
	headers = append(headers, "label")
	row = append(row, "0")


	file, err := os.Create(file_name)

	if err != nil {
		fmt.Println("Error creating csv file for test image")
		panic(err)
	}

	csvwriter := csv.NewWriter(file)
	csvwriter.Write(headers)
	// csvwriter.WriteAll(data)
	csvwriter.Write(row)
	csvwriter.Flush()

	file.Close()
}


// Converts an image to a 48*48 grayscale image, in the form of an array of pixels
func imageToArray(image_path string) []int {
	src, err := imaging.Open(image_path)
	if err != nil {
		fmt.Println("Error opening image path")
		panic(err)
	}

	// convert to grayscale
	src = imaging.Grayscale(src)
	// resize to fit the model input
	src = imaging.Resize(src, 48, 48, imaging.Box)

	err = imaging.Save(src, "image_test/out.jpg")

	if err != nil {
		fmt.Println("failed to save image: ", err)
	} 

	return(getPixelArray(src))
}


// Convters the given image to an array of pixels. Assumes the array is already grayscale
func getPixelArray(img image.Image) []int {
	var pixels []int

	// get width/height of image
	dimensions := img.Bounds()
	width, height := dimensions.Max.X, dimensions.Max.Y

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// rgba values are all the same; get the r value as pixel value
			r, _, _, _ := img.At(x, y).RGBA()
			pixelVal := r / 257
			// need to divide by 257 to scale in range [0, 255]
			pixels = append(pixels, int(pixelVal))
		}
	}

	return pixels
}


// Given a label, map to the corresponding emotion
func get_emotion_from_label(label int) string {
	switch (label) {
	case 0:
		return "angry"
	case 1:
		return "disgust"
	case 2:
		return "fear"
	case 3:
		return "happy"
	case 4:
		return "neutral"
	case 5:
		return "sad"
	case 6:
		return "surprise"
	default:
		return ""
	}
}
