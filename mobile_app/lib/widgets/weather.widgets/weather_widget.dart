import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:intl/intl.dart';
import 'package:mobile_app/models/weatherModel.dart';
import 'package:mobile_app/widgets/weather.widgets/weather_info.dart';
import '../../services/weatherService.dart';
import '../../utils/constants.dart';

class ExpandableWeatherCard extends StatefulWidget {
  const ExpandableWeatherCard({super.key});

  @override
  State<ExpandableWeatherCard> createState() => _ExpandableWeatherCardState();
}

class _ExpandableWeatherCardState extends State<ExpandableWeatherCard> {
  // API key for OpenWeatherMap
  final _weatherService = WeatherService(ApiConstants.OPENWEATHERMAP_API_KEY);
  Weather? _weather;

  // fetch weather data
  _fetchWeatherData() async {
    // Get the current city
    final cityName = await _weatherService.getCurrentCity();

    // Get weather data for the current city
    try {
      final weather = await _weatherService.getCurrentWeather(cityName);
      setState(() {
        _weather = weather;
      });
    } catch (e) {
      print('Error: $e');
    }
  }

  @override
  void initState() {
    super.initState();

    // Fetch weather on startup
    _fetchWeatherData();
  }

  String getFormattedDate() {
    DateTime now = DateTime.now();
    String formattedDate = DateFormat('EEE, MMM d').format(now);
    return 'Today, $formattedDate';
  }

  bool isExpanded = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: GestureDetector(
        onTap: () {
          setState(() {
            isExpanded = !isExpanded;
          });
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
          decoration: BoxDecoration(
            color: const Color(0xFFF8FAFC),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            children: [
              Row(
                children: [
                  Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.amber.shade100,
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(10.0),
                      child: SvgPicture.asset(
                        'assets/icons/lucide_sun.svg',
                        height: 24,
                        width: 24,
                        color: Colors.amber,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        getFormattedDate(),
                        style: TextStyle(
                          fontSize: 14,
                          color: Color.fromRGBO(100, 116, 139, 1),
                        ),
                      ),
                      Text(
                        _weather != null
                            ? '${_weather?.temperature.round()}°C'
                            : 'Loading Temperature...',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          const Icon(Icons.location_on,
                              size: 16, color: Colors.red),
                          const SizedBox(width: 4),
                          Text(
                            _weather?.cityName ??
                                "Loading City...", // Replace with dynamic location
                            style: const TextStyle(
                              fontSize: 14,
                              color: Color.fromRGBO(100, 116, 139, 1),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  const Spacer(),
                  Icon(
                    isExpanded ? Icons.expand_less : Icons.expand_more,
                    size: 28,
                  ),
                ],
              ),

              // Expanded Content
              AnimatedSize(
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeInOut,
                  child: isExpanded
                      ? Column(
                          children: [
                            const SizedBox(height: 8),
                            const Divider(
                              color: Color.fromRGBO(226, 232, 240, 1),
                            ),
                            const SizedBox(height: 8),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceAround,
                              children: [
                                WeatherInfo(
                                  title: "UV Index",
                                  value: "${_weather?.uvIndex ?? 0}",
                                  iconPath: 'assets/icons/lucide_sun.svg',
                                  iconColor: Colors.amber,
                                ),
                                WeatherInfo(
                                  title: "Humidity",
                                  value: "${_weather?.humidity ?? 0}%",
                                  iconPath: 'assets/icons/lucide_droplets.svg',
                                  iconColor: Colors.blue,
                                ),
                                WeatherInfo(
                                  title: "Wind Speed",
                                  value: "${_weather?.windSpeed ?? 0} km/h",
                                  iconPath: 'assets/icons/lucide_cloud.svg',
                                  iconColor: Colors.blueGrey,
                                ),
                              ],
                            ),
                          ],
                        )
                      : const SizedBox())
            ],
          ),
        ),
      ),
    );
  }
}
