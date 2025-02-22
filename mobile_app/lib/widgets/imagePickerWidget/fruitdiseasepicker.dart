import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class FruitDiseasePicker extends StatefulWidget {
  const FruitDiseasePicker({super.key});

  @override
  _FruitDiseasePickerState createState() => _FruitDiseasePickerState();
}

class _FruitDiseasePickerState extends State<FruitDiseasePicker> {
  File? _image;
  String? _category;
  String? _predictionLabel;
  double? _confidence;
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _resetPrediction();
      });
    }
  }

  void _resetPrediction() {
    _category = null;
    _predictionLabel = null;
    _confidence = null;
  }

  Future<void> _predictDisease() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
      _resetPrediction();
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://10.0.2.2:8000/predict'),
      );
      request.files.add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var jsonResponse = jsonDecode(await response.stream.bytesToString());
        setState(() {
          _category = jsonResponse['EfficientNet']['label'];
          _predictionLabel = jsonResponse['DenseNet']['label'];
          _confidence = jsonResponse['DenseNet']['confidence'];
        });
      } else {
        _handleError("Error predicting disease");
      }
    } catch (e) {
      _handleError("Network error");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _handleError(String message) {
    setState(() {
      _category = message;
      _predictionLabel = message;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Fruit Disease Picker"),
        backgroundColor: Colors.green,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Text(
                "Fruit Disease Detection",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 20),

              // Display larger image
              _image != null
                  ? Image.file(_image!, height: 350)
                  : const Icon(Icons.image, size: 200, color: Colors.grey),
              const SizedBox(height: 20),

              // Buttons for picking image
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text("Capture"),
                  ),
                  const SizedBox(width: 10),
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.image),
                    label: const Text("Gallery"),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              // Predict button
              ElevatedButton.icon(
                onPressed: _isLoading ? null : _predictDisease,
                icon: const Icon(Icons.search),
                label: const Text("Predict"),
              ),
              const SizedBox(height: 20),

              // Show loading or prediction results
              _isLoading
                  ? const CircularProgressIndicator()
                  : _category != null
                      ? PredictionCard(
                          category: _category!,
                          predictionLabel: _predictionLabel!,
                          confidence: _confidence,
                        )
                      : Container(),
            ],
          ),
        ),
      ),
    );
  }
}

class PredictionCard extends StatelessWidget {
  final String category;
  final String predictionLabel;
  final double? confidence;

  const PredictionCard({
    super.key,
    required this.category,
    required this.predictionLabel,
    this.confidence,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      color: Colors.grey[100],
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Prediction Results",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Text("• Category: $category", style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 5),
            Text("• Prediction: $predictionLabel",
                style: const TextStyle(fontSize: 16)),
            if (confidence != null) ...[
              const SizedBox(height: 10),
              const Text("Confidence:", style: TextStyle(fontSize: 16)),
              const SizedBox(height: 5),
              LinearProgressIndicator(
                value: confidence! / 100,
                backgroundColor: Colors.grey[300],
                color: confidence! > 75 ? Colors.green : Colors.orange,
                minHeight: 10,
              ),
              const SizedBox(height: 5),
              Text("${confidence!.toStringAsFixed(2)}%",
                  style: const TextStyle(fontSize: 16)),
            ],
          ],
        ),
      ),
    );
  }
}
