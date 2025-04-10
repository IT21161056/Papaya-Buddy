import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mobile_app/models/predictionModel.dart';
import 'package:mobile_app/utils/constants.dart'; // For MediaType

class HistoryService {
  static Future<void> savePrediction({
    required String userId,
    required String diseaseId,
    required File imageFile, // Pass the File object directly
  }) async {
    final String url = '${BaseURL.BASE_URL}/api/v1/history/create_history';
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
            return 'application/octet-stream';
        }
      }

      // Get the content type for the file
      final contentType = getContentType(imageFile.path);

      request.files.add(
        await http.MultipartFile.fromPath(
          'uploaded_img', // Field name for the file
          imageFile.path,
          contentType:
              MediaType.parse(contentType), // Set the correct MediaType
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
          print(
              "Failed to save prediction history: ${responseBody['message']}");
        }
      } else {
        print("Failed to save prediction history: ${response.statusCode}");
      }
    } catch (error) {
      print("Error saving prediction history: $error");
    }
  }

  static Future<List<Prediction>> getDiagnosis({required String userId}) async {
    final String url =
        '${BaseURL.BASE_URL}/api/v1/history/get_user_history/$userId';

    try {
      final response = await http.get(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final dynamic decodedData = json.decode(response.body);

        if (decodedData is List) {
          return decodedData.map((data) => Prediction.fromJson(data)).toList();
        } else {
          print(
              "Unexpected JSON format. Expected List but got ${decodedData.runtimeType}");
          return [];
        }
      } else {
        print(
            "Failed to load diagnosis history. Status code: ${response.statusCode}");
        return [];
      }
    } catch (error) {
      print("Error fetching diagnosis history: $error");
      return [];
    }
  }
}
