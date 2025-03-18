import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/models/maturityModel.dart';
import 'package:mobile_app/services/maturityServices.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';
import 'package:mobile_app/views/maturityView/maturityView.dart';
import '../../views/maturityView/PapayaMaturityInfoScreen.dart';

class MaturityScreen extends StatefulWidget {
  @override
  _MaturityScreenState createState() => _MaturityScreenState();
}

class _MaturityScreenState extends State<MaturityScreen> {
  Uint8List? _imageBytes;
  File? _image;
  String? _result;
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;

  final MaturityStageDisplayModel dummyMaturityStageDisplay =
      MaturityStageDisplayModel(
    maturityStage: MaturityStage(
        id: "3",
        stage: "Mature",
        description:
            "The fruit has reached its typical color. Skin yields slightly to pressure and has a sweet aroma. The flesh is juicy and flavorful.",
        timeToReach: "3–4 months after fruit set",
        timeGapToNextStage: "3–7 days",
        bestTimeToHarvest: "Ideal time to harvest for best flavor and texture",
        suggestedImageUrls: [
          'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvabyEg9ytOEh1IEBVRfIe3Vl4gRmit4HcOQ&s'
        ]),
    imageFile: null, // Replace with actual File object when needed
  );

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
      _result = null;
    });
  }

  // Function to get ripeness color based on the prediction
  Color _getRipenessColor(String ripeness) {
    switch (ripeness.toLowerCase()) {
      case "not_mature":
        return Colors.teal;
      case "partially_mature":
        return Colors.lightGreen;
      case "mature":
        return Colors.green;
      case "rotten":
        return Colors.yellowAccent;
      default:
        return Colors.grey;
    }
  }

  String _getStage(String ripeness) {
    switch (ripeness.toLowerCase()) {
      case "not_mature":
        return 'not mature';
      case "partially_mature":
        return "partially mature";
      case "mature":
        return 'mature';
      case "rotten":
        return 'rotten';
      default:
        return '';
    }
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
        Uri.parse(
          'http://127.0.0.1:5000/predict', // Added trailing slash to match your FastAPI endpoint
        ),
      );

      request.files
          .add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var jsonResponse = jsonDecode(await response.stream.bytesToString());
        setState(() {
          _result = jsonResponse['predicted_class'];
          _isLoading = false;
        });

        var stage = _getStage(_result ?? '');

        MaturityStage? data = await MaturityServices.getMaturityData(stage);

        if (data != null) {
          if (mounted) {
            MaturityStageDisplayModel maturityDisplay =
                MaturityStageDisplayModel(
              maturityStage: data,
              imageFile:
                  _image ?? null, // Replace with actual image file if available
            );

            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) =>
                    MaturityView(maturityStage: dummyMaturityStageDisplay),
              ),
            );
          }
        } else {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("No data found for $stage")));
          }
        }
      } else {
        throw Exception("Error predicting disease: ${response.statusCode}");
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Error predicting disease')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Colors.white,
        // appBar: AppBar(
        //   backgroundColor: Colors.white,
        //   elevation: 0,
        //   leading: GestureDetector(
        //     onTap: () {
        //       Navigator.pop(context);
        //     },
        //     child: Container(
        //       margin: const EdgeInsets.all(10),
        //       decoration: BoxDecoration(
        //         color: Colors.grey.shade200,
        //         shape: BoxShape.circle,
        //       ),
        //       child: const Icon(
        //         Icons.arrow_back_ios_new,
        //         color: Colors.black,
        //         size: 16,
        //       ),
        //     ),
        //   ),
        //   title: const Text(
        //     "Maturity Stage Detection",
        //     style: TextStyle(
        //       fontWeight: FontWeight.bold,
        //       color: Colors.black,
        //       fontSize: 20,
        //     ),
        //   ),
        //   centerTitle: true,
        // ),
        body: SafeArea(
            child: Column(
          children: [
            // header
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
                    'Maturity Stage Detection',
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
                        // onPressed: _isLoading ? null : _predictDisease,
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => MaturityView(
                                  maturityStage: dummyMaturityStageDisplay),
                            ),
                          );
                        },
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
                                    SvgPicture.asset(
                                      'assets/icons/magic.svg',
                                      height: 16,
                                      width: 16,
                                      color: const Color.fromARGB(
                                          255, 150, 215, 255),
                                    ),
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
