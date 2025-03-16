import 'dart:io';

class MaturityStage {
  final String stage;
  final String description;
  final String timeToReach;
  final String timeGapToNextStage;
  final String bestTimeToHarvest;
  final List<String> suggestedImageUrls;
  String? id; // For MongoDB ObjectId

  // Valid stages enumeration
  static const List<String> validStages = [
    "Not Mature",
    "Partially Mature",
    "Mature",
    "Rotten"
  ];

  // Constructor
  MaturityStage({
    required this.stage,
    required this.description,
    required this.timeToReach,
    required this.timeGapToNextStage,
    required this.bestTimeToHarvest,
    required this.suggestedImageUrls,
    this.id,
  }) : assert(validStages.contains(stage), 'Invalid stage value');

  // Create from JSON (when fetching from API)
  factory MaturityStage.fromJson(Map<String, dynamic> json) {
    return MaturityStage(
      id: json['_id'],
      stage: json['stage'],
      description: json['description'],
      timeToReach: json['timeToReach'],
      timeGapToNextStage: json['timeGapToNextStage'],
      bestTimeToHarvest: json['bestTimeToHarvest'],
      suggestedImageUrls: json['suggestedImageUrls'],
    );
  }
}

class MaturityStageDisplayModel {
  final MaturityStage maturityStage;
  final File? imageFile;

  MaturityStageDisplayModel({
    required this.maturityStage,
    this.imageFile,
  });

  String? get id => maturityStage.id;
  String get stage => maturityStage.stage;
  String get description => maturityStage.description;
  String get timeToReach => maturityStage.timeToReach;
  String get timeGapToNextStage => maturityStage.timeGapToNextStage;
  String get bestTimeToHarvest => maturityStage.bestTimeToHarvest;
  List<String> get suggestedImageUrls => maturityStage.suggestedImageUrls;

  MaturityStageDisplayModel copyWith({
    MaturityStage? maturityStage,
    File? imageFile,
  }) {
    return MaturityStageDisplayModel(
      maturityStage: maturityStage ?? this.maturityStage,
      imageFile: imageFile ?? this.imageFile,
    );
  }
}
