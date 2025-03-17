import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:mobile_app/models/treatmentModel.dart';

class TreatmentService {
  static const String baseUrl = "http://192.168.1.4:5080/treatment";

  static Future<List<Treatment>?> getTreatmentDataByDisease(
      String diseaseId) async {
    final String url =
        'http://192.168.1.4:5080/api/v1/treatment/by-disease/67d831e2feacec6a44777146';
    try {
      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);

        if (responseData['success'] == true &&
            responseData['data'] != null &&
            responseData['data'].isNotEmpty) {
          // Parse the array of treatments and return as a List
          List<Treatment> treatments = [];
          for (var treatmentData in responseData['data']) {
            treatments.add(Treatment.fromJson(treatmentData));
          }
          return treatments;
        } else {
          print("No data found for treatments");
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
