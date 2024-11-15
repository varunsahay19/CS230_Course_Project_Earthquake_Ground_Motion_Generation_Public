package main

// (c) 2024 Patrick Nieman
// Constructs a series of indexes representing the rock types that an earthquake passes through between source and site for a particular record

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

var pairs = []line{}
var rockTypes = []int{}
var polys = []poly{}
var holeReferences = []int{}
var pointsPerTick = 3
var setCount = true
var pointsPerPath = 20
var lastFound = -1

// Utility for reading from CSV files
func readMatrix(filePath string) [][]float64 {
	matrix := [][]float64{}
	ph, _ := os.Open(filePath)
	defer ph.Close()
	h := bufio.NewReader(ph)
	for {
		line, e := h.ReadString('\n')
		if e != nil {
			break
		}
		vals := strings.Split(strings.ReplaceAll(line, "\n", ""), ",")
		row := []float64{}
		for _, s := range vals {
			v, _ := strconv.ParseFloat(s, 64)
			row = append(row, v)
		}
		matrix = append(matrix, row)
	}
	ph.Close()
	return matrix
}

// Determines which polygon and thus rock corresponds to a particular coordinate point
func getRockTypes(pointCoords []float64) []int {
	indices := []int{}
	indexMap := map[int]int{}
	valid := []bool{}
	p := point{pointCoords[0], pointCoords[1], pointCoords[2]}

	// Optimization
	if lastFound >= 0 && insidePoly(p, polys[lastFound]) {
		return []int{lastFound}
	}

	// Call geometry analysis library
	for i, poly := range polys {
		if insidePoly(p, poly) {
			indices = append(indices, i)
			if holeReferences[i] == -1 && (i == len(holeReferences)-1 || holeReferences[i+1] == -1) {
				return indices
			}
			valid = append(valid, true)
			indexMap[i] = len(valid) - 1
		}
	}

	// Handle holes in polygons
	for i, index := range indices {
		if holeReferences[index] != -1 {
			valid[i] = false
			valid[indexMap[holeReferences[index]]] = false
		}
	}
	validIndices := []int{}
	for i, index := range indices {
		if valid[i] {
			validIndices = append(validIndices, index)
		}
	}
	return validIndices
}

func main() {
	// Load source and site coordiantes for each record
	for _, row := range readMatrix("coordinates.csv") {
		pairs = append(pairs, _line(point{row[1], row[0], 0.0}, point{row[3], row[2], 0.0}))
	}

	// Load polygon data and corresponding rock types
	for _, row := range readMatrix("rockTypes.csv") {
		rockTypes = append(rockTypes, int(row[0]))
	}

	for _, row := range readMatrix("holeReferences.csv") {
		holeReferences = append(holeReferences, int(row[0]))
	}

	for _, row := range readMatrix("polys.csv") {
		points := []point{}
		for i := 0; i < len(row)-2; i += 2 {
			points = append(points, point{row[i], row[i+1], 0.0})
		}
		polys = append(polys, _poly(points).windCC(false))
	}

	// Compile source-to-site path for specified number of points or for a number of points proportional to distance
	tic := time.Now()
	f, _ := os.Create("pathData.csv")
	for i, l := range pairs {
		incr := 1.0
		if !setCount {
			incr = 1.0 / math.Floor((l.length*float64(pointsPerTick))+1)
		} else {
			incr = 1.0 / (float64(pointsPerPath) - 1)
		}
		for p := 0.0; p < 1.0001; p += incr {
			types := getRockTypes(l.pEval(p))
			if len(types) != 1 {
				fmt.Println(i, p, len(types))
			} else {
				lastFound = types[0]
				if p+incr < 1.0001 {
					f.WriteString(fmt.Sprintf("%v,", rockTypes[types[0]]))
				} else {
					f.WriteString(fmt.Sprintf("%v", rockTypes[types[0]]))
				}
			}
		}
		f.WriteString("\n")
		fmt.Println(i)
		if i == 10 {
			break
		}
	}
	fmt.Println(time.Since(tic))
	f.Close()

}
