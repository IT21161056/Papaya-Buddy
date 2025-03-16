import 'dart:io';

class Disease {
  final String id;
  final String name;
  final String affectedArea;
  final List<String> symptoms;
  final String diseaseType;
  final String description;
  final String preventiveMeasures;
  final List<String> suggestedImageUrls;
  final DateTime createdAt;

  Disease(
      {required this.id,
      required this.name,
      required this.affectedArea,
      required this.symptoms,
      required this.diseaseType,
      required this.description,
      required this.preventiveMeasures,
      required this.createdAt,
      required this.suggestedImageUrls});

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
        suggestedImageUrls: List<String>.from(json['suggested_image_urls']));
  }
}

class DiseaseDisplayModel {
  final Disease disease;
  final File? imageFile;

  DiseaseDisplayModel({
    required this.disease,
    this.imageFile,
  });

  // Helper getters to access disease properties directly
  String get id => disease.id;
  String get name => disease.name;
  String get affectedArea => disease.affectedArea;
  List<String> get symptoms => disease.symptoms;
  String get diseaseType => disease.diseaseType;
  String get description => disease.description;
  String get preventiveMeasures => disease.preventiveMeasures;
  DateTime get createdAt => disease.createdAt;
  List<String> get suggestedImageUrls => disease.suggestedImageUrls;

  // Create a copy with optional changes
  DiseaseDisplayModel copyWith({
    Disease? disease,
    File? imageFile,
  }) {
    return DiseaseDisplayModel(
      disease: disease ?? this.disease,
      imageFile: imageFile ?? this.imageFile,
    );
  }
}
