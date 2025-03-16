import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';
import 'package:mobile_app/services/diseaseService.dart';
import 'package:mobile_app/models/diseaseModel.dart';

class FruitDiseasePicker extends StatefulWidget {
  const FruitDiseasePicker({super.key});

  @override
  _FruitDiseasePickerState createState() => _FruitDiseasePickerState();
}

class _FruitDiseasePickerState extends State<FruitDiseasePicker> {
  File? _image;
  String? disease_prediction;
  bool _isLoading = false;
  bool _isImageLoading = false;
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
    setState(() {
      disease_prediction = null;
    });
  }

  Future<void> _predictDisease() async {
    if (_image == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please select an image first")),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _resetPrediction();
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://10.0.2.2:5000/predict'),
      );
      request.files
          .add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var jsonResponse = jsonDecode(await response.stream.bytesToString());
        String diseaseName =
            jsonResponse['disease_prediction']; // Extract disease name

        Disease? data = await DiseaseService.getDiseaseData(diseaseName);

        if (data != null) {
          if (mounted) {
            DiseaseDisplayModel diseaseDisplay = DiseaseDisplayModel(
              disease: data,
              imageFile: null, // Replace with actual image file if available
            );

            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => DiseaseView(disease: diseaseDisplay),
              ),
            );
          }
        } else {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text("No data found for $diseaseName")),
            );
          }
        }
      } else {
        _handleError("Error predicting disease");
      }
    } catch (e) {
      print("Exception details: $e");
      _handleError("Network error: ${e.toString()}");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _handleError(String message) {
    if (mounted) {
      setState(() {
        disease_prediction = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
    }
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
