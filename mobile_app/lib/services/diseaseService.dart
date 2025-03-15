import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:mobile_app/models/diseaseModel.dart';

class DiseaseService {
  static const String baseUrl = "http://192.168.187.155:5080/disease";

  static Future<Disease?> getDiseaseData(String diseaseName) async {
    final String url =
        'http://192.168.187.155:5080/api/v1/disease?name=$diseaseName';
    print(diseaseName);
    try {
      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);

        if (responseData['success'] == true &&
            responseData['data'] != null &&
            responseData['data'].isNotEmpty) {
          return Disease.fromJson(responseData['data'][0]);
        } else {
          print("No data found for disease: $diseaseName");
          return null;
        }
      } else {
        print("Failed to load data: ${response.statusCode}");
        return null;
      }
    } catch (error) {
      print("Error fetching disease data: $error");
      return null;
    }
  }
}
