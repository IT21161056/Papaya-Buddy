import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';
import 'package:mobile_app/views/healthyView/healthy_view.dart';
import 'package:mobile_app/services/diseaseService.dart';
import 'package:mobile_app/models/diseaseModel.dart';

class ImagePickerPage extends StatefulWidget {
  const ImagePickerPage({super.key});

  @override
  _ImagePickerPageState createState() => _ImagePickerPageState();
}

class _ImagePickerPageState extends State<ImagePickerPage> {
  File? _image;
  String? _category;
  String? _predictionLabel;
  double? _confidence;
  bool _isLoading = false;
  bool _isImageLoading = false; // New variable to track image loading state
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _isImageLoading = true; // Start loading
      });

      // Simulate a delay for image processing (replace with actual logic if needed)
      await Future.delayed(Duration(seconds: 2)); // Simulate a 2-second delay

      setState(() {
        _image = File(pickedFile.path);
        _isImageLoading = false; // Stop loading
        _resetPrediction();
      });
    }
  }

  void _resetPrediction() {
    setState(() {
      _category = null;
      _predictionLabel = null;
      _confidence = null;
    });
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
      request.files
          .add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var jsonResponse = jsonDecode(await response.stream.bytesToString());
        String diseaseName =
            jsonResponse['DenseNet']['label']; // Extract disease name

        Disease? diseaseData = await DiseaseService.getDiseaseData(diseaseName);

        if (diseaseData != null) {
          // Create DiseaseDisplayModel
          DiseaseDisplayModel diseaseDisplay = DiseaseDisplayModel(
            disease: diseaseData,
            imageFile: null, // Pass the selected image file
          );

          // Navigate to the appropriate view based on the disease name
          if (diseaseName == "Healthy Leaf" || diseaseName == "Healthy Fruit") {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => HealthyView(disease: diseaseDisplay),
              ),
            );
          } else {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => DiseaseView(disease: diseaseDisplay),
              ),
            );
          }
        } else {
          _handleError("No data found for $diseaseName");
        }
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
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: GestureDetector(
          onTap: () {
            Navigator.pop(context);
          },
          child: Container(
            margin: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.grey.shade200,
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.arrow_back_ios_new,
              color: Colors.black,
              size: 16,
            ),
          ),
        ),
        title: const Text(
          "Disease Detection",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.black,
            fontSize: 20,
          ),
        ),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Image Placeholder
            Container(
              width: double.infinity,
              height: 290,
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(20),
              ),
              child: _isImageLoading
                  ? Center(
                      child: CircularProgressIndicator(
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
                      ),
                    )
                  : _image == null
                      ? Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SvgPicture.asset(
                              'assets/icons/capture.svg',
                              height: 20,
                              width: 40,
                              color: Colors.grey.shade400,
                            ),
                            const SizedBox(height: 5),
                            Text(
                              "No image selected",
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.grey.shade600,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              "Take a photo or choose from gallery",
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey.shade500,
                              ),
                            ),
                          ],
                        )
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: Image.file(
                            _image!,
                            fit: BoxFit.cover,
                          ),
                        ),
            ),
            const SizedBox(height: 10),
            // Image Selection Options
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                children: [
                  GestureDetector(
                    onTap: () => _pickImage(ImageSource.camera),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: const Color(0xFFDDEEFF),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: SvgPicture.asset(
                            'assets/icons/camera.svg',
                            height: 20,
                            width: 24,
                            color: const Color(0xFF1A73E8),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              "Take Photo",
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              "Use your camera to capture the disease",
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey.shade600,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Divider(),
                  GestureDetector(
                    onTap: () => _pickImage(ImageSource.gallery),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: const Color(0xFFE5F8E6),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: SvgPicture.asset(
                            'assets/icons/gallery.svg',
                            height: 24,
                            width: 24,
                            color: const Color(0xFF23C55E),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              "Choose from Gallery",
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              "Select an existing photo from your device",
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey.shade600,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            // Tips for Better Detection
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Tips for better detection",
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  _buildTip(1, "Ensure good lighting conditions"),
                  _buildTip(2, "Keep the camera steady and focused"),
                  _buildTip(3, "Capture the affected area clearly"),
                ],
              ),
            ),
            const SizedBox(height: 20),
            // Predict Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _predictDisease,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color.fromRGBO(37, 100, 235, 1),
                  padding: EdgeInsets.symmetric(vertical: 20),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  elevation: 0,
                ),
                child: AnimatedSwitcher(
                  duration: Duration(milliseconds: 300), // Smooth transition
                  child: _isLoading
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor:
                                    AlwaysStoppedAnimation<Color>(Colors.white),
                              ),
                            ),
                            SizedBox(
                                width:
                                    10), // Add spacing between loader and text
                            Text(
                              "Predicting...",
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              "Predict",
                              style: TextStyle(
                                color: const Color.fromARGB(255, 150, 215, 255),
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            SizedBox(width: 6),
                            SvgPicture.asset(
                              'assets/icons/magic.svg',
                              height: 16,
                              width: 16,
                              color: const Color.fromARGB(255, 150, 215, 255),
                            ),
                          ],
                        ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTip(int number, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(
            width: 24,
            decoration: BoxDecoration(
              color: Colors.blue.shade100,
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                number.toString(),
                style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: const Color.fromARGB(255, 176, 176, 176)),
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 14,
                color: const Color.fromARGB(255, 162, 161, 161),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
