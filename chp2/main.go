package main

import (
	"fmt"
	t "gorgonia.org/tensor"
)

func main() {
	fmt.Println("Hello")

	inputs := t.New(t.WithBacking([]float32{1.0, 2.0, 3.0, 2.5}))

	rawWeights := []float32{
		0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}
	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))

	biases := t.New(t.WithBacking([]float32{2.0, 3, 0.5}))

	dP, _ := t.Dot(weights, inputs)
	output, _ := t.Add(dP, biases)

	fmt.Println(output)

}
