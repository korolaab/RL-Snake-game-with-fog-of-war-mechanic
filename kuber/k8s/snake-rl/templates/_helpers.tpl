{{/*
Expand the name of the chart.
*/}}
{{- define "snake-rl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "snake-rl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "snake-rl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "snake-rl.labels" -}}
helm.sh/chart: {{ include "snake-rl.chart" . }}
{{ include "snake-rl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "snake-rl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "snake-rl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Generate ISO datetime - use from values if provided, otherwise generate dynamically
*/}}
{{- define "myapp.runID" -}}
{{- if .Values.experiment.deploymentTimeISO }}
{{- .Values.experiment.deploymentTimeISO }}
{{- else }}
{{- now | date "2006-01-02T15:04:05Z07:00" }}
{{- end }}
{{- end }}

{{/*
Environment variables for RabbitMQ
*/}}
{{- define "snake-rl.rabbitmqEnvVars" -}}
- name: ENABLE_RABBITMQ
  value: {{ .Values.rabbitmq.enabled | quote }}
- name: RABBITMQ_HOST
  value: {{ .Values.rabbitmq.host | quote }}
- name: RABBITMQ_PORT
  value: {{ .Values.rabbitmq.port | quote }}
- name: RABBITMQ_USERNAME
  value: {{ .Values.rabbitmq.username | quote }}
- name: RABBITMQ_PASSWORD
  value: {{ .Values.rabbitmq.password | quote }}
- name: RABBITMQ_VHOST
  value: {{ .Values.rabbitmq.vhost | quote }}
- name: RABBITMQ_EXCHANGE
  value: {{ .Values.rabbitmq.exchange | quote }}
- name: RABBITMQ_ROUTING_KEY
  value: {{ .Values.rabbitmq.routingKey | quote }}
- name: RABBITMQ_EXCHANGE_TYPE
  value: {{ .Values.rabbitmq.exchangeType | quote }}
- name: RABBITMQ_DURABLE
  value: {{ .Values.rabbitmq.durable | quote }}
- name: RABBITMQ_ASYNC
  value: {{ .Values.rabbitmq.async | quote }}
- name: RABBITMQ_CONNECTION_TIMEOUT
  value: {{ .Values.rabbitmq.connectionTimeout | quote }}
- name: RABBITMQ_RETRY_DELAY
  value: {{ .Values.rabbitmq.retryDelay | quote }}
- name: RABBITMQ_MAX_RETRIES
  value: {{ .Values.rabbitmq.maxRetries | quote }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "snake-rl.commonEnvVars" -}}
- name: PYTHONUNBUFFERED
  value: "1"
- name: EXPERIMENT_NAME
  value: {{ .Values.experiment.name | quote }}
- name: RUN_ID
  value: {{ include "myapp.runID" . | quote }}
- name: LOG_LEVEL
  value: {{ .Values.logging.level | quote }}
- name: LOG_FILE
  value: {{ .Values.logging.file | quote }}
- name: ENABLE_CONSOLE_LOGS
  value: {{ .Values.logging.enableConsoleLogs | quote }}
{{- end }}

