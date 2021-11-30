package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"

	// "github.com/sjwhitworth/golearn/knn"

	// "github.com/sjwhitworth/golearn/linear_models"

	"github.com/sjwhitworth/golearn/ensemble"

	"github.com/sjwhitworth/golearn/evaluation"
)

func main() {
	// Load train and test data from the csv files
	fmt.Print("Loading data...")

	train, error := base.ParseCSVToInstances("../data/mnist/mnist_train.csv", true)
	if error != nil {
		panic(error)
	}

	test, error := base.ParseCSVToInstances("../data/mnist/mnist_test.csv", true)
	if error != nil {
		panic(error)
	}

	fmt.Println("Done")

	// // Create the model using a linear support vector machine
	// model, error := linear_models.NewLinearSVC("l1", "l2", true, 1.0, 1e-4)

	model := ensemble.NewRandomForest(64, 28*28)

	// if error != nil {
	// 	panic(error)
	// }

	// suppress output
	base.Silent()

	fmt.Print("Training model...")
	// train the model
	model.Fit(train)

	fmt.Println("Done.")

	// fmt.Println("Loading model...")

	// model := linear_models.NewModel()
	// model.Load("mnist")

	// Test the model using the test data
	fmt.Println("Testing model...")

	predictions, error := model.Predict(test)
	if error != nil {
		panic(error)
	}

	// get the confusion matrix based on the predictions
	confusion, error := evaluation.GetConfusionMatrix(test, predictions)
	if error != nil {
		panic(error)
	}

	fmt.Println(evaluation.GetSummary(confusion))
	model.Save("forest")
}
