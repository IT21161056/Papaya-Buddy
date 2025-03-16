import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

class MaturityGuideScreen extends StatelessWidget {
  const MaturityGuideScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      body: SafeArea(
        child: Column(
          children: [
            // Header
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
                    'Papaya Maturity Guide',
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

            // Content
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  // Introduction
                  IntroCard().animate().fadeIn(delay: 200.ms, duration: 300.ms),

                  // Maturity Stages
                  ...List.generate(
                    maturityData.length,
                    (index) => StageCard(
                      stageInfo: maturityData[index],
                      index: index,
                    ).animate().fadeIn(
                          delay: (300 + (index * 100)).ms,
                          duration: 300.ms,
                        ),
                  ),

                  // Learn More Button
                  LearnMoreButton()
                      .animate()
                      .fadeIn(delay: 700.ms, duration: 300.ms),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class IntroCard extends StatelessWidget {
  const IntroCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      margin: const EdgeInsets.only(bottom: 24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          Text(
            'Understanding Papaya Maturity',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1A1A1A),
            ),
          ),
          SizedBox(height: 12),
          Text(
            'Learn about the different stages of papaya maturity to ensure optimal harvesting and consumption. Each stage has unique characteristics and best uses.',
            style: TextStyle(
              fontSize: 16,
              color: Color(0xFF64748B),
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }
}

class StageCard extends StatelessWidget {
  final Map<String, dynamic> stageInfo;
  final int index;

  const StageCard({
    required this.stageInfo,
    required this.index,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final Color stageColor = Color(
        int.parse(stageInfo['color'].substring(1), radix: 16) + 0xFF000000);

    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Stage Header
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              border: Border(
                bottom: BorderSide(
                  color: const Color(0xFFF1F5F9),
                  width: 1,
                ),
              ),
            ),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: stageColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                stageInfo['stage'],
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: stageColor,
                ),
              ),
            ),
          ),

          // Stage Content
          Container(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Description
                Text(
                  stageInfo['description'],
                  style: const TextStyle(
                    fontSize: 15,
                    color: Color(0xFF1A1A1A),
                    height: 1.6,
                  ),
                ),
                const SizedBox(height: 24),

                // Timeline Container
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF8FAFC),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    children: [
                      // Time to Reach
                      Row(
                        children: [
                          Container(
                            width: 40,
                            height: 40,
                            margin: const EdgeInsets.only(right: 16),
                            decoration: BoxDecoration(
                              color: stageColor.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Icon(
                              Icons.calendar_today,
                              size: 20,
                              color: stageColor,
                            ),
                          ),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Time to Reach',
                                  style: TextStyle(
                                    fontSize: 14,
                                    color: Color(0xFF64748B),
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  stageInfo['timeToReach'],
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w500,
                                    color: Color(0xFF1A1A1A),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),

                      // Time to Next Stage
                      Row(
                        children: [
                          Container(
                            width: 40,
                            height: 40,
                            margin: const EdgeInsets.only(right: 16),
                            decoration: BoxDecoration(
                              color: stageColor.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Icon(
                              Icons.schedule,
                              size: 20,
                              color: stageColor,
                            ),
                          ),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Time to Next Stage',
                                  style: TextStyle(
                                    fontSize: 14,
                                    color: Color(0xFF64748B),
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  stageInfo['timeGapToNextStage'],
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w500,
                                    color: Color(0xFF1A1A1A),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),

                // Harvest Info
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF8FAFC),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 40,
                        height: 40,
                        margin: const EdgeInsets.only(right: 16),
                        decoration: BoxDecoration(
                          color: stageColor.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Icon(
                          Icons.error_outline,
                          size: 20,
                          color: stageColor,
                        ),
                      ),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Best Time to Harvest',
                              style: TextStyle(
                                fontSize: 14,
                                color: Color(0xFF64748B),
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              stageInfo['bestTimeToHarvest'],
                              style: const TextStyle(
                                fontSize: 16,
                                color: Color(0xFF1A1A1A),
                                height: 1.5,
                              ),
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
        ],
      ),
    );
  }
}

class LearnMoreButton extends StatelessWidget {
  const LearnMoreButton({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 40),
      child: Material(
        color: const Color(0xFFEFF6FF),
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          onTap: () {},
          borderRadius: BorderRadius.circular(12),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                Text(
                  'Learn More About Papaya Care',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF2563EB),
                  ),
                ),
                SizedBox(width: 8),
                Icon(
                  Icons.arrow_forward,
                  size: 20,
                  color: Color(0xFF2563EB),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

final List<Map<String, dynamic>> maturityData = [
  {
    'stage': 'Not Mature',
    'color': '#22c55e',
    'description':
        'The fruit is fully green, hard, and underdeveloped. At this stage, the fruit has not yet developed its characteristic sweetness and remains firm to the touch. The outer skin is completely green, and the flesh is white or pale green. It is not suitable for raw consumption but may be used for pickling or cooking in certain dishes.',
    'timeToReach': '0–2 months after fruit set',
    'timeGapToNextStage': '1–2 weeks',
    'bestTimeToHarvest':
        'Not ideal for harvesting unless intended for cooking or pickling.'
  },
  {
    'stage': 'Partially Mature',
    'color': '#84cc16',
    'description':
        'The fruit begins to transition from its green state, showing slight yellowing on the surface. The texture remains firm, but the flesh inside starts to soften slightly, developing mild sweetness. While not yet fully ripe, the fruit can be harvested at this stage and allowed to ripen further off the tree. This stage is ideal for transport and short-term storage before consumption.',
    'timeToReach': '2–3 months after fruit set',
    'timeGapToNextStage': '3–7 days',
    'bestTimeToHarvest':
        'Suitable for harvesting if you want to store the fruit for a few days before consumption.'
  },
  {
    'stage': 'Mature',
    'color': '#eab308',
    'description':
        'The fruit is at its peak ripeness, with yellow-orange skin and a soft, smooth texture. The flesh inside is juicy, sweet, and rich in flavor. At this stage, the papaya is perfect for eating fresh, blending into smoothies, or using in fruit salads. The natural sugars are at their highest, making the fruit highly palatable. It can also be refrigerated for a few days to extend its shelf life.',
    'timeToReach': '3–4 months after fruit set',
    'timeGapToNextStage': '2–4 days',
    'bestTimeToHarvest':
        'The optimal time to harvest is when the fruit has 25–50% yellow coloration on its skin.'
  },
  {
    'stage': 'Rotten',
    'color': '#ef4444',
    'description':
        'The fruit becomes excessively soft, with visible dark spots, blemishes, or mold growth. The flesh starts to break down, releasing an overripe, sometimes fermented aroma. The taste changes, losing its sweetness and developing an unpleasant flavor. At this stage, the fruit is no longer suitable for fresh consumption but may be used for composting, animal feed, or baking applications where overripe fruit is acceptable.',
    'timeToReach': '4–5 months after fruit set',
    'timeGapToNextStage': 'N/A',
    'bestTimeToHarvest':
        'Not suitable for harvesting. Overripe papayas are best used in baking or discarded if too far gone.'
  },
  {
    'stage': 'Default',
    'color': '#6b7280',
    'description':
        'This stage is used when the maturity level of the papaya fruit cannot be determined or does not match any of the defined stages. The fruit may exhibit irregular characteristics, such as uneven ripening, damage, or disease, making it difficult to classify. It is recommended to inspect the fruit carefully before deciding on its use or disposal.',
    'timeToReach': 'N/A',
    'timeGapToNextStage': 'N/A',
    'bestTimeToHarvest': 'N/A'
  }
];
