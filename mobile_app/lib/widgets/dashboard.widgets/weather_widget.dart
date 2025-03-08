import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/widgets/dashboard.widgets/weather_info.dart';

class ExpandableWeatherCard extends StatefulWidget {
  const ExpandableWeatherCard({super.key});

  @override
  State<ExpandableWeatherCard> createState() => _ExpandableWeatherCardState();
}

class _ExpandableWeatherCardState extends State<ExpandableWeatherCard> {
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
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "Today, Feb 26",
                        style: TextStyle(
                          fontSize: 14,
                          color: Color.fromRGBO(100, 116, 139, 1),
                        ),
                      ),
                      Text(
                        "24°C / 25°C",
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
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
                            const Row(
                              mainAxisAlignment: MainAxisAlignment.spaceAround,
                              children: [
                                WeatherInfo(
                                  title: "UV Index",
                                  value: "High",
                                  iconPath: 'assets/icons/lucide_sun.svg',
                                  iconColor: Colors.amber,
                                ),
                                WeatherInfo(
                                  title: "Humidity",
                                  value: "68%",
                                  iconPath: 'assets/icons/lucide_droplets.svg',
                                  iconColor: Colors.blue,
                                ),
                                WeatherInfo(
                                  title: "Cloud Cover",
                                  value: "15%",
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
