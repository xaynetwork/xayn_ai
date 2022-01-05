// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'oneshot.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Sender _$SenderFromJson(Map json) => Sender(
      const MessagePortConverter().fromJson(json['port'] as MessagePort?),
    );

Map<String, dynamic> _$SenderToJson(Sender instance) => <String, dynamic>{
      'port': const MessagePortConverter().toJson(instance.port),
    };
