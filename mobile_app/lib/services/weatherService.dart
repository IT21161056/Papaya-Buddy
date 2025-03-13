import 'dart:convert';
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import '../models/weatherModel.dart';
import '../utils/constants.dart';

class WeatherService {
  String API_KEY = ApiConstants.OPENWEATHERMAP_API_KEY;
  static const BASE_URL = ApiConstants.OPENWEATHERMAP_BASE_URL;

  WeatherService(this.API_KEY);

  Future<Weather> getCurrentWeather(String cityName) async {
    final response = await http.get(
      Uri.parse('$BASE_URL/weather?q=$cityName&appid=$API_KEY&units=metric'),
    );

    if (response.statusCode == 200) {
      final jsonData = json.decode(response.body);

      // Get coordinates for One Call API (for UV index)
      final lat = jsonData['coord']['lat'];
      final lon = jsonData['coord']['lon'];

      // Get additional data from One Call API
      final oneCallResponse = await http.get(
        Uri.parse(
            '$BASE_URL/onecall?lat=$lat&lon=$lon&exclude=minutely,hourly,daily,alerts&appid=$API_KEY&units=metric'),
      );

      if (oneCallResponse.statusCode == 200) {
        final oneCallData = json.decode(oneCallResponse.body);

        // Add UV index to the original data
        jsonData['uvi'] = oneCallData['current']['uvi'];
      }

      return Weather.fromJson(jsonData);
    } else {
      throw Exception(
          'Failed to load weather data. Status code: ${response.statusCode}');
    }
  }

  Future<String> getCurrentCity() async {
    try {
      // Request location permission
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.deniedForever) {
        return 'Permission Denied';
      }

      // Get current position
      Position position = await Geolocator.getCurrentPosition(
          desiredAccuracy: LocationAccuracy.high);

      // Convert location into a list of placemarks
      List<Placemark> placemarks =
          await placemarkFromCoordinates(position.latitude, position.longitude);

      if (placemarks.isNotEmpty && placemarks[0].locality != null) {
        return placemarks[0].locality!;
      } else {
        return 'Unknown';
      }
    } catch (e) {
      print("Error fetching city: $e");
      return 'Unknown';
    }
  }
}
