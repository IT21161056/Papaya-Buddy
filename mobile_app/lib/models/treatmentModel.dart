class Treatment {
  final String id;
  final String method;
  final String description;
  final String diseaseId;
  final List<String> treatmentType;
  final List<String> effectiveness;
  final String sideEffects;
  final String precautions;
  final DateTime createdAt;

  Treatment({
    required this.id,
    required this.method,
    required this.description,
    required this.diseaseId,
    required this.treatmentType,
    required this.effectiveness,
    required this.sideEffects,
    required this.precautions,
    required this.createdAt,
  });

  // Create Treatment instance from a map
  factory Treatment.fromJson(Map<String, dynamic> json) {
    return Treatment(
      id: json['_id'],
      method: json['method'],
      description: json['description'],
      diseaseId: json['diseaseId'],
      treatmentType: List<String>.from(json['treatment_type']),
      effectiveness: List<String>.from(json['effectiveness']),
      sideEffects: json['side_effects'],
      precautions: json['precautions'],
      createdAt: DateTime.parse(json["created_at"]),
    );
  }
}

class TreatmentDisplayModel {
  final Treatment treatment;

  TreatmentDisplayModel({
    required this.treatment,
  });

  // Helper getters to access treatment properties directly
  String get id => treatment.id;
  String get method => treatment.method;
  String get description => treatment.description;
  String get diseaseId => treatment.diseaseId;
  List<String> get treatmentType => treatment.treatmentType;
  List<String> get effectiveness => treatment.effectiveness;
  String get sideEffects => treatment.sideEffects;
  String get precautions => treatment.precautions;
  DateTime get createdAt => treatment.createdAt;

  // Create a copy with optional changes
  TreatmentDisplayModel copyWith({
    Treatment? treatment,
  }) {
    return TreatmentDisplayModel(
      treatment: treatment ?? this.treatment,
    );
  }
}
