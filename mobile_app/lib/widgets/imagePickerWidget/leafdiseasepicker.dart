import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mobile_app/utils/constants.dart';
import 'package:mobile_app/models/diseaseModel.dart';
import 'package:mobile_app/services/diseaseService.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';
import 'package:mobile_app/views/healthyView/healthy_view.dart';

class LeafDiseasePicker extends StatefulWidget {
  const LeafDiseasePicker({super.key});

  @override
  _LeafDiseasePickerState createState() => _LeafDiseasePickerState();
}

class _LeafDiseasePickerState extends State<LeafDiseasePicker> {
  File? _image;
  String _healthStatus = '';
  String _disease = '';
  double _confidence = 0;
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
    setState(() {
      _disease = '';
      _healthStatus = '';
      _confidence = 0;
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
        Uri.parse('${BaseURL.BASE_URL}/predict-leaf-disease'),
      );

      request.files
          .add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var jsonResponse = jsonDecode(await response.stream.bytesToString());

        _healthStatus = jsonResponse['health_status'];
        _confidence = jsonResponse['confidence'];

        // Check if the leaf is healthy or has disease
        if (_healthStatus == "Healthy") {
          Disease? data = await DiseaseService.getDiseaseData('healthy leaf');
          if (data != null) {
            if (mounted) {
              DiseaseDisplayModel diseaseDisplay = DiseaseDisplayModel(
                disease: data,
                imageFile: _image,
              );

              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => HealthyView(disease: diseaseDisplay),
                ),
              );
            }
          } else {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("No data found for $_healthStatus")),
              );
            }
          }
        } else {
          // For unhealthy leaves, get the disease
          _disease = jsonResponse['disease'];

          if (_disease != null) {
            Disease? data = await DiseaseService.getDiseaseData(_disease);

            if (data != null) {
              if (mounted) {
                DiseaseDisplayModel diseaseDisplay = DiseaseDisplayModel(
                  disease: data,
                  imageFile: _image,
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
                  SnackBar(content: Text("No data found for $_disease")),
                );
              }
            }
          } else {
            _handleError("Disease information missing in the response");
          }
        }
      } else {
        _handleError("Error predicting disease: ${response.statusCode}");
      }
    } catch (e) {
      _handleError("Network error: ${e.toString()}");
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _handleError(String message) {
    if (mounted) {
      setState(() {
        _healthStatus = "";
        _disease = "";
        _confidence = 0;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Colors.white,
        body: SafeArea(
            child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
              decoration: const BoxDecoration(
                color: Colors.white,
                border: Border(
                  bottom: BorderSide(
                    color: Color(0xFFF1F5F9),
                    width: 1,
                  ),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Material(
                    color: const Color(0xFFF8FAFC),
                    borderRadius: BorderRadius.circular(12),
                    child: InkWell(
                      onTap: () => Navigator.pop(context),
                      borderRadius: BorderRadius.circular(12),
                      child: Container(
                        width: 40,
                        height: 40,
                        alignment: Alignment.center,
                        child: const Icon(
                          Icons.chevron_left,
                          size: 24,
                          color: Color(0xFF1A1A1A),
                        ),
                      ),
                    ),
                  ),
                  const Text(
                    'Leaf Disease Detection',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1A1A1A),
                    ),
                  ),
                  const SizedBox(width: 40),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Image Placeholder
                    Container(
                      width: double.infinity,
                      height: 290,
                      decoration: BoxDecoration(
                        color: AppColors.background,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: _image == null
                          ? Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                SvgPicture.asset(
                                  'assets/icons/capture.svg',
                                  height: 40,
                                  width: 40,
                                  color:
                                      AppColors.textSecondary.withOpacity(0.5),
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
                                    color: AppColors.textSecondary
                                        .withOpacity(0.5),
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
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
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
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ],
                                  ),
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
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
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
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ],
                                  ),
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
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF8FAFC),
                        borderRadius: BorderRadius.circular(12),
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
                          const SizedBox(height: 12),
                          ...List.generate(
                            3,
                            (index) {
                              final tips = [
                                'Ensure good lighting conditions',
                                'Keep the camera steady and focused',
                                'Capture the affected area clearly',
                              ];

                              return Padding(
                                padding: const EdgeInsets.only(bottom: 12),
                                child: Row(
                                  children: [
                                    Container(
                                      width: 24,
                                      height: 24,
                                      decoration: BoxDecoration(
                                        color: const Color(0xFFE0F2FE),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      alignment: Alignment.center,
                                      child: Text(
                                        '${index + 1}',
                                        style: TextStyle(
                                          fontSize: 14,
                                          fontWeight: FontWeight.w600,
                                          color: Color(0xFF0284C7),
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Text(
                                        tips[index],
                                        style: const TextStyle(
                                          fontSize: 14,
                                          color: Color(0xFF1A1A1A),
                                          height: 1.4,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
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
                          padding: const EdgeInsets.symmetric(vertical: 20),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          elevation: 0,
                        ),
                        child: AnimatedSwitcher(
                          duration: const Duration(
                              milliseconds: 300), // Smooth transition
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
                                            AlwaysStoppedAnimation<Color>(
                                                Colors.white),
                                      ),
                                    ),
                                    const SizedBox(width: 10),
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
                                        color: Colors.white,
                                        fontSize: 16,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                    const SizedBox(width: 6),
                                    SvgPicture.asset('assets/icons/magic.svg',
                                        height: 16,
                                        width: 16,
                                        color: Colors.white)
                                  ],
                                ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
          ],
        )));
  }
}
