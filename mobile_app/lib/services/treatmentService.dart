import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:mobile_app/models/treatmentModel.dart';

class DiseaseService {
  static const String baseUrl = "http://192.168.8.193:5080/treatment";

  static Future<Treatment?> getTreatmentData(String treatmentName) async {
    final String url =
        'http://192.168.8.193:5080/api/v1/treatment?name=$treatmentName';
    try {
      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);

        if (responseData['success'] == true &&
            responseData['data'] != null &&
            responseData['data'].isNotEmpty) {
          return Treatment.fromJson(responseData['data'][0]);
        } else {
          print("No data found for treatments: $treatmentName");
          return null;
        }
      } else {
        print("Failed to load data: ${response.statusCode}");
        return null;
      }
    } catch (error) {
      print("Error fetching treatment data: $error");
      return null;
    }
  }
}
