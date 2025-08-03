"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Upload, Brain, RefreshCw, Activity, Zap, List } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from "recharts"

export default function MLPipelineDashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [retrainProgress, setRetrainProgress] = useState(0)
  const [modelStatus, setModelStatus] = useState("online")
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [availableClasses, setAvailableClasses] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const bulkUploadRef = useRef<HTMLInputElement>(null)

  // Fetch available classes on component mount
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch("http://localhost:8000/model/info")
        if (!response.ok) {
          throw new Error("Failed to fetch model info")
        }
        const data = await response.json()
        if (data.model_info && data.model_info.class_names) {
          setAvailableClasses(data.model_info.class_names)
        }
      } catch (error) {
        console.error("Error fetching model info:", error)
        // Optionally set an error state or default classes
      }
    }
    fetchModelInfo()
  }, [])

  // Handle paste event for images
  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const items = event.clipboardData?.items
      if (items) {
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf("image") !== -1) {
            const blob = items[i].getAsFile()
            if (blob) {
              const pastedFile = new File([blob], `pasted_image_${Date.now()}.png`, { type: blob.type })
              setSelectedFile(pastedFile)
              setPrediction(null)
              event.preventDefault() // Prevent default paste behavior (e.g., pasting into a text field)
              break
            }
          }
        }
      }
    }

    document.addEventListener("paste", handlePaste)

    return () => {
      document.removeEventListener("paste", handlePaste)
    }
  }, [])

  // Mock data for visualizations (kept for demonstration)
  const modelPerformanceData = [
    { epoch: 1, accuracy: 0.65, loss: 0.89 },
    { epoch: 2, accuracy: 0.72, loss: 0.76 },
    { epoch: 3, accuracy: 0.78, loss: 0.65 },
    { epoch: 4, accuracy: 0.83, loss: 0.54 },
    { epoch: 5, accuracy: 0.87, loss: 0.43 },
    { epoch: 6, accuracy: 0.89, loss: 0.38 },
    { epoch: 7, accuracy: 0.91, loss: 0.32 },
    { epoch: 8, accuracy: 0.92, loss: 0.29 },
    { epoch: 9, accuracy: 0.93, loss: 0.26 },
    { epoch: 10, accuracy: 0.94, loss: 0.23 },
  ]

  const classDistributionData = [
    { name: "Airplane", value: 1000, color: "#8884d8" },
    { name: "Automobile", value: 1000, color: "#82ca9d" },
    { name: "Bird", value: 1000, color: "#ffc658" },
    { name: "Cat", value: 1000, color: "#ff7300" },
    { name: "Deer", value: 1000, color: "#00ff00" },
    { name: "Dog", value: 1000, color: "#ff0000" },
    { name: "Frog", value: 1000, color: "#0000ff" },
    { name: "Horse", value: 1000, color: "#ff00ff" },
    { name: "Ship", value: 1000, color: "#00ffff" },
    { name: "Truck", value: 1000, color: "#ffff00" },
  ]

  const requestLatencyData = [
    { containers: 1, latency: 245, throughput: 12 },
    { containers: 2, latency: 156, throughput: 23 },
    { containers: 4, latency: 89, throughput: 45 },
    { containers: 8, latency: 67, throughput: 78 },
    { containers: 16, latency: 52, throughput: 124 },
  ]

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setPrediction(null)
    }
  }

  const handleBulkUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setUploadedFiles((prev) => [...prev, ...files])
  }

  const handlePredict = async () => {
    if (!selectedFile) return

    setIsLoading(true)
    setPrediction(null) // Clear previous prediction

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Prediction failed")
      }

      const result = await response.json()
      let finalPrediction = result.prediction

      // Apply "unknown" logic if confidence is below 50%
      if (finalPrediction.confidence < 0.5) {
        finalPrediction = {
          ...finalPrediction,
          predicted_class: "Unknown",
          message: "Confidence below 50%. Predicted as Unknown.",
        }
      }

      setPrediction(finalPrediction) // Set the prediction from the API response
    } catch (error: any) {
      // Explicitly type error as any for message property
      console.error("Error during prediction:", error)
      alert(`Prediction failed: ${error.message || error}`)
      setPrediction({
        predicted_class: "Error",
        confidence: 0,
        all_probabilities: {}, // Ensure this is an empty object
        message: error.message || "An unknown error occurred.",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleRetrain = async () => {
    if (uploadedFiles.length === 0) {
      alert("Please upload training data first")
      return
    }

    setRetrainProgress(0)
    setIsLoading(true)

    try {
      // Step 1: Upload files to the backend
      setRetrainProgress(10)
      const uploadFormData = new FormData()
      uploadedFiles.forEach((file) => {
        uploadFormData.append("files", file)
      })

      const uploadResponse = await fetch("http://localhost:8000/upload/data", {
        method: "POST",
        body: uploadFormData,
      })

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json()
        throw new Error(errorData.detail || "Failed to upload training data")
      }

      const uploadResult = await uploadResponse.json()
      const batchId = String(uploadResult.batch_id) // Explicitly cast to string for URL parameter

      if (!batchId) {
        throw new Error("Batch ID not received from upload. Retraining cannot proceed.")
      }
      console.log("Batch ID received for retraining:", batchId) // Log the batchId

      setRetrainProgress(30) // Update progress after upload
      // Step 2: Trigger retraining with the actual batch_id
      const retrainResponse = await fetch(`http://localhost:8000/retrain?batch_id=${batchId}`, {
        method: "POST",
      })

      if (!retrainResponse.ok) {
        const errorData = await retrainResponse.json()
        throw new Error(errorData.detail || "Retraining failed to trigger")
      }

      // Poll for retraining status
      const pollInterval = setInterval(async () => {
        const statusResponse = await fetch("http://localhost:8000/retrain/status")
        const statusData = await statusResponse.json()

        // Adjust progress to reflect overall process (upload + retraining)
        // Assuming retraining itself is 70% of the progress (30% for upload + 70% for retraining)
        setRetrainProgress(30 + statusData.progress * 0.7)

        if (statusData.status === "completed" || statusData.status === "failed") {
          clearInterval(pollInterval)
          setIsLoading(false)
          if (statusData.status === "completed") {
            alert("Model retrained successfully!")
          } else {
            alert(`Model retraining failed: ${statusData.message}`)
          }
          setRetrainProgress(0) // Reset progress bar
          setUploadedFiles([]) // Clear uploaded files after retraining
        }
      }, 1000) // Poll every 1 second
    } catch (error: any) {
      // Explicitly type error as any
      console.error("Error during retraining:", error)
      alert(`Retraining failed: ${error.message || error}`)
      setIsLoading(false)
      setRetrainProgress(0)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900">Lesly Predictor</h1>
          <p className="text-lg text-gray-600">CIFAR-10 Image Classification System</p>
          <div className="flex justify-center items-center gap-4">
            <Badge variant={modelStatus === "online" ? "default" : "destructive"} className="flex items-center gap-1">
              <Activity className="w-3 h-3" />
              Model Status: {modelStatus}
            </Badge>
            <Badge variant="outline" className="flex items-center gap-1">
              <Zap className="w-3 h-3" />
              Uptime: 99.9%
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="predict" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="predict">Prediction</TabsTrigger>
            <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
            <TabsTrigger value="retrain">Retraining</TabsTrigger>
            <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          </TabsList>

          {/* Prediction Tab */}
          <TabsContent value="predict" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Upload Image
                  </CardTitle>
                  <CardDescription>Upload an image for classification (CIFAR-10 classes)</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    <Input
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      ref={fileInputRef}
                      className="hidden"
                    />
                    <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="mb-4">
                      <Upload className="w-4 h-4 mr-2" />
                      Choose Image
                    </Button>
                    {selectedFile ? (
                      <div className="space-y-2">
                        <p className="text-sm text-gray-600">Selected: {selectedFile.name}</p>
                        <img
                          src={URL.createObjectURL(selectedFile) || "/placeholder.svg"}
                          alt="Selected"
                          className="max-w-full h-48 object-contain mx-auto rounded"
                        />
                      </div>
                    ) : (
                      <p className="text-gray-500 text-sm">Or paste an image (Ctrl+V / Cmd+V)</p>
                    )}
                  </div>
                  <Button onClick={handlePredict} disabled={!selectedFile || isLoading} className="w-full">
                    <Brain className="w-4 h-4 mr-2" />
                    {isLoading ? "Predicting..." : "Predict"}
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Prediction Results</CardTitle>
                  <CardDescription>Model prediction and confidence scores</CardDescription>
                </CardHeader>
                <CardContent>
                  {prediction ? (
                    <div className="space-y-4">
                      <Alert>
                        <AlertDescription>
                          <strong>Predicted Class:</strong> {prediction.predicted_class} (
                          {(prediction.confidence * 100).toFixed(1)}% confidence)
                        </AlertDescription>
                      </Alert>
                      {prediction.message && prediction.predicted_class === "Unknown" && (
                        <Alert variant="warning" className="mt-2">
                          <AlertDescription>{prediction.message}</AlertDescription>
                        </Alert>
                      )}
                      <div className="space-y-2">
                        <h4 className="font-semibold">All Class Probabilities:</h4>
                        {prediction.all_probabilities &&
                          Object.entries(prediction.all_probabilities)
                            .sort(([, probA], [, probB]) => (probB as number) - (probA as number))
                            .map(([className, prob]) => (
                              <div key={className} className="flex justify-between items-center">
                                <span>{className}</span>
                                <div className="flex items-center gap-2 w-32">
                                  <Progress value={(prob as number) * 100} className="flex-1" />
                                  <span className="text-sm">{((prob as number) * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            ))}
                        {prediction.message && prediction.predicted_class !== "Unknown" && (
                          <Alert variant="destructive">
                            <AlertDescription>{prediction.message}</AlertDescription>
                          </Alert>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-500 text-center py-8">Upload an image and click predict to see results</p>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* New Card for Available Classes */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <List className="w-5 h-5" />
                  Predictable Classes
                </CardTitle>
                <CardDescription>The Lesly Predictor can classify images into these categories:</CardDescription>
              </CardHeader>
              <CardContent>
                {availableClasses.length > 0 ? (
                  <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 text-sm text-gray-700">
                    {availableClasses.map((className, index) => (
                      <li key={index} className="flex items-center">
                        <span className="mr-1 text-blue-500">•</span> {className}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-500">Loading available classes...</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Visualizations Tab */}
          <TabsContent value="visualizations" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Model Training Performance</CardTitle>
                  <CardDescription>Accuracy and loss over training epochs</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={modelPerformanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#8884d8" strokeWidth={2} />
                      <Line type="monotone" dataKey="loss" stroke="#82ca9d" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Class Distribution</CardTitle>
                  <CardDescription>Distribution of classes in CIFAR-10 dataset</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={classDistributionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {classDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Load Testing Results</CardTitle>
                  <CardDescription>Latency and throughput vs number of containers</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={requestLatencyData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="containers" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="latency" fill="#8884d8" name="Latency (ms)" />
                      <Bar yAxisId="right" dataKey="throughput" fill="#82ca9d" name="Throughput (req/s)" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Retraining Tab */}
          <TabsContent value="retrain" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Upload Training Data
                  </CardTitle>
                  <CardDescription>Upload multiple images for model retraining</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    <Input
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={handleBulkUpload}
                      ref={bulkUploadRef}
                      className="hidden"
                    />
                    <Button variant="outline" onClick={() => bulkUploadRef.current?.click()} className="mb-4">
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Images
                    </Button>
                    <p className="text-sm text-gray-600">{uploadedFiles.length} files uploaded</p>
                  </div>
                  {uploadedFiles.length > 0 && (
                    <div className="max-h-32 overflow-y-auto">
                      {uploadedFiles.map((file, index) => (
                        <div key={index} className="text-sm text-gray-600 py-1">
                          {file.name}
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <RefreshCw className="w-5 h-5" />
                    Trigger Retraining
                  </CardTitle>
                  <CardDescription>Retrain the model with uploaded data</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Alert>
                    <AlertDescription>
                      Retraining will use the uploaded images to improve model performance. This process may take
                      several minutes.
                    </AlertDescription>
                  </Alert>

                  {retrainProgress > 0 && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Training Progress</span>
                        <span>{retrainProgress.toFixed(1)}%</span>
                      </div>
                      <Progress value={retrainProgress} />
                    </div>
                  )}

                  <Button
                    onClick={handleRetrain}
                    disabled={uploadedFiles.length === 0 || retrainProgress > 0}
                    className="w-full"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Start Retraining
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Monitoring Tab */}
          <TabsContent value="monitoring" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Model Accuracy</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-600">94.2%</div>
                  <p className="text-sm text-gray-600">Current test accuracy</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Avg Response Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-600">156ms</div>
                  <p className="text-sm text-gray-600">Last 24 hours</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Requests Today</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-purple-600">2,847</div>
                  <p className="text-sm text-gray-600">Successful predictions</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>System Health</CardTitle>
                <CardDescription>Real-time monitoring of system components</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>CPU Usage</span>
                      <span>45%</span>
                    </div>
                    <Progress value={45} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Memory Usage</span>
                      <span>67%</span>
                    </div>
                    <Progress value={67} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>GPU Usage</span>
                      <span>23%</span>
                    </div>
                    <Progress value={23} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Disk Usage</span>
                      <span>34%</span>
                    </div>
                    <Progress value={34} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
