class Disease {
  final String id;
  final String name;
  final String affectedArea;
  final List<String> symptoms;
  final String diseaseType;
  final String description;
  final String preventiveMeasures;
  final DateTime createdAt;

  Disease({
    required this.id,
    required this.name,
    required this.affectedArea,
    required this.symptoms,
    required this.diseaseType,
    required this.description,
    required this.preventiveMeasures,
    required this.createdAt,
  });

  factory Disease.fromJson(Map<String, dynamic> json) {
    return Disease(
      id: json["_id"],
      name: json["name"],
      affectedArea: json["affected_area"],
      symptoms: List<String>.from(json["symptoms"]),
      diseaseType: json["disease_type"],
      description: json["description"],
      preventiveMeasures: json["preventive_measures"],
      createdAt: DateTime.parse(json["created_at"]),
    );
  }
}
