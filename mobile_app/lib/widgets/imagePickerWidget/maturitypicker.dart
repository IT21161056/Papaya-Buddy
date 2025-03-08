import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../../views/maturityView/PapayaMaturityInfoScreen.dart';

class MaturityScreen extends StatefulWidget {
  @override
  _MaturityScreenState createState() => _MaturityScreenState();
}

class _MaturityScreenState extends State<MaturityScreen> {
  Uint8List? _imageBytes;
  String? _result;
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;

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

  Future<void> _uploadImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      Uint8List imageBytes = await image.readAsBytes();
      setState(() {
        _imageBytes = imageBytes;
        _result = null; // Reset result when a new image is picked
        _isLoading = true;
      });

      var request = http.MultipartRequest(
        'POST',
        Uri.parse(
          'http://127.0.0.1:5000/predict',
        ), // Replace with your backend URL
      );

      request.files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: 'image.jpg'),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();

      setState(() {
        _result = jsonDecode(responseData)['predicted_class'];
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Papaya Maturity",
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        leading: Padding(
          padding: EdgeInsets.all(8.0),
          child: Image.asset("assets/Papaya.png"), // Add an icon for branding
        ),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // const Text(
            //   "Fruit Maturity Detection",
            //   style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            // ),
            // const SizedBox(height: 20),

            // Display image
            _imageBytes != null
                ? Image.memory(_imageBytes!, height: 200)
                : const Icon(Icons.image, size: 150, color: Colors.grey),
            const SizedBox(height: 20),

            // Buttons for picking image
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _uploadImage(),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Capture"),
                ),
                const SizedBox(width: 10),
                ElevatedButton.icon(
                  onPressed: () => _uploadImage(),
                  icon: const Icon(Icons.image),
                  label: const Text("Gallery"),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Predict button
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _uploadImage,
              icon: const Icon(Icons.search),
              label: const Text("Predict"),
            ),
            const SizedBox(height: 20),

            // Show loading or prediction results
            _isLoading
                ? const CircularProgressIndicator()
                : _result != null
                    ? Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: _getRipenessColor(_result!),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          _result!,
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      )
                    : Container(),

            const SizedBox(height: 30),

            // Ripeness Levels Legend
            Column(
              children: [
                const Text(
                  "Ripeness Levels",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 10),
                _buildLegend("Not Ripe", Colors.teal),
                _buildLegend("Partially Ripe", Colors.lightGreen),
                _buildLegend("Ripe", Colors.green),
                _buildLegend("Rotten", Colors.yellowAccent),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => PapayaMaturityInfoScreen(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                  ),
                  child: const Text(
                    "Learn About Maturity Levels",
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegend(String label, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(width: 20, height: 10, color: color),
          const SizedBox(width: 10),
          Text(label, style: const TextStyle(fontSize: 14)),
        ],
      ),
    );
  }
}
