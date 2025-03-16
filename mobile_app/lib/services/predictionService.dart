import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart'; // For MediaType

class SavePredictionHistory {
  static const String baseUrl = "http://10.0.2.2:5080";

  static Future<void> savePrediction({
    required String userId,
    required String diseaseId,
    required File imageFile, // Pass the File object directly
  }) async {
    final String url = '$baseUrl/api/v1/history/create_history';
    try {
      // Create a multipart request
      final request = http.MultipartRequest('POST', Uri.parse(url));

      request.fields['userId'] = userId;
      request.fields['diseaseId'] = diseaseId;

      // Determine the content type dynamically
      String getContentType(String filePath) {
        final extension = filePath.split('.').last.toLowerCase();
        switch (extension) {
          case 'jpg':
          case 'jpeg':
            return 'image/jpeg';
          case 'png':
            return 'image/png';
          case 'gif':
            return 'image/gif';
          default:
            return 'application/octet-stream'; // Fallback for unknown types
        }
      }

      // Get the content type for the file
      final contentType = getContentType(imageFile.path);

      request.files.add(
        await http.MultipartFile.fromPath(
          'uploaded_img', // Field name for the file
          imageFile.path,
          contentType: MediaType.parse(contentType), // Set the correct MediaType
        ),
      );

      // Send the request
      final response = await request.send();

      if (response.statusCode == 201) {
        final responseData = await response.stream.bytesToString();
        final Map<String, dynamic> responseBody = jsonDecode(responseData);

        if (responseBody['success'] == true) {
          print("Prediction history saved successfully!");
        } else {
          print("Failed to save prediction history: ${responseBody['message']}");
        }
      } else {
        print("Failed to save prediction history: ${response.statusCode}");
      }
    } catch (error) {
      print("Error saving prediction history: $error");
    }
  }
}