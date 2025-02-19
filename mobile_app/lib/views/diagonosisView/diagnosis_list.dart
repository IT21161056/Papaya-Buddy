import 'package:flutter/material.dart';
import 'package:mobile_app/views/diagonosisView/disease_details.dart';


class DiagnosisList extends StatelessWidget {
  const DiagnosisList({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.only(left: 8.0, bottom: 8.0),
          child: Text(
            "Your Diagnosis",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: pastDiagnoses.length,
            itemBuilder: (context, index) {
              return DiagnosisListItem(
                title: pastDiagnoses[index]['title'],
                date: pastDiagnoses[index]['date'],
                result: pastDiagnoses[index]['result'],
                onDetailsPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => DiseaseDetailsPage(
                        diseaseName: pastDiagnoses[index]['title'] ??
                            'No Title Available', // Default value if null
                        description: pastDiagnoses[index]['description'] ??
                            'No description available', // Default value if null
                        remedy: pastDiagnoses[index]['remedy'] ??
                            'No remedy available', // Default value if null
                        images: pastDiagnoses[index]['images'] ??
                            [], // Ensure this is a list, even if empty
                      ),
                    ),
                  );
                },
              );
            },
          ),
        ),
      ],
    );
  }
}

final List<Map<String, dynamic>> pastDiagnoses = [
  {
    'title': 'Mite Disease on Papaya',
    'date': 'Feb 12, 2025',
    'result': 'High Risk',
    'description': 'Mites cause leaf discoloration and stunted growth.',
    'remedy': 'Use neem oil or sulfur-based sprays. Introduce predatory mites.',
    'images': [
      'assets/bg.jpg',
      'assets/bg.jpg'
    ],
  },
  {
    'title': 'Black Spot Fungus',
    'date': 'Jan 28, 2025',
    'result': 'Moderate Risk',
    'description':
        'Black spots on leaves lead to defoliation and reduced yield.',
    'remedy': 'Apply copper-based fungicides and improve air circulation.',
    'images': [
      'assets/fox.jpg',
      'assets/c.jpg'
    ],
  },
];

class DiagnosisListItem extends StatelessWidget {
  final String title;
  final String date;
  final String result;
  final VoidCallback onDetailsPressed;

  const DiagnosisListItem({
    super.key,
    required this.title,
    required this.date,
    required this.result,
    required this.onDetailsPressed,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      margin: const EdgeInsets.symmetric(vertical: 4.0, horizontal: 8.0),
      child: ListTile(
        leading: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: SizedBox(
            width: 50, // Same width as before
            height: 50, // Reduced height for the image
            child: Image.asset(
              'assets/bg.jpg', // Correct path to your local image
              fit: BoxFit.cover,
            ),
          ),
        ),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text("$date"),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              result,
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: result == "High Risk"
                    ? Colors.red
                    : result == "Moderate Risk"
                        ? Colors.orange
                        : Colors.green,
              ),
            ),
            const SizedBox(width: 8),
            IconButton(
              onPressed: onDetailsPressed,
              icon: const Icon(Icons.arrow_forward, color: Colors.green),
              tooltip: "View Details",
            ),
          ],
        ),
      ),
    );
  }
}


