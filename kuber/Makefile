.PHONY: help init deploy destroy status logs clean passwords connect urls

# Configuration
NAMESPACE := data-stack
ENVIRONMENT ?= dev
CLICKHOUSE_RELEASE := clickhouse
RABBITMQ_RELEASE := rabbitmq

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RED := \033[0;31m
NC := \033[0m

help: ## Show this help
	@echo "ClickHouse Data Stack Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  1. make init      # Create config files from templates"
	@echo "  2. make secrets   # Generate and create secrets"
	@echo "  3. make deploy    # Deploy ClickHouse + RabbitMQ"
	@echo "  4. make urls      # Get connection URLs"

init: ## Initialize configuration files from templates
	@echo "$(YELLOW)🔧 Initializing configuration files...$(NC)"
	@if [ ! -f k8s/data-stack/clickhouse/values.yaml ]; then \
		cp k8s/data-stack/clickhouse/values-template.yaml k8s/data-stack/clickhouse/values.yaml; \
		echo "$(GREEN)✅ Created ClickHouse values.yaml$(NC)"; \
	else \
		echo "$(BLUE)ℹ️  ClickHouse values.yaml already exists$(NC)"; \
	fi
	@if [ ! -f k8s/data-stack/rabbitmq/values.yaml ]; then \
		cp k8s/data-stack/rabbitmq/values-template.yaml k8s/data-stack/rabbitmq/values.yaml; \
		echo "$(GREEN)✅ Created RabbitMQ values.yaml$(NC)"; \
	else \
		echo "$(BLUE)ℹ️  RabbitMQ values.yaml already exists$(NC)"; \
	fi
	@if [ ! -f scripts/deploy.sh ]; then \
		cp scripts/deploy-template.sh scripts/deploy.sh; \
		chmod +x scripts/deploy.sh; \
		echo "$(GREEN)✅ Created deploy.sh$(NC)"; \
	fi
	@if [ ! -f scripts/port-forward.sh ]; then \
		cp scripts/port-forward-template.sh scripts/port-forward.sh; \
		chmod +x scripts/port-forward.sh; \
		echo "$(GREEN)✅ Created port-forward.sh$(NC)"; \
	fi

generate-passwords: ## Generate secure passwords and update values files
	@echo "$(YELLOW)🔐 Generating secure passwords...$(NC)"
	@mkdir -p .secrets
	@if [ ! -f .secrets/clickhouse_password ]; then \
		openssl rand -base64 20 | tr -d "=+/" | cut -c1-16 > .secrets/clickhouse_password; \
		echo "$(GREEN)✅ Generated ClickHouse password$(NC)"; \
	else \
		echo "$(BLUE)ℹ️  ClickHouse password already exists$(NC)"; \
	fi
	@if [ ! -f .secrets/rabbitmq_password ]; then \
		openssl rand -base64 20 | tr -d "=+/" | cut -c1-16 > .secrets/rabbitmq_password; \
		echo "$(GREEN)✅ Generated RabbitMQ password$(NC)"; \
	else \
		echo "$(BLUE)ℹ️  RabbitMQ password already exists$(NC)"; \
	fi
	@echo "$(BLUE)📁 Passwords saved in .secrets/ directory$(NC)"

create-secrets: generate-passwords ## Create Kubernetes secrets from password files
	@echo "$(YELLOW)🔐 Creating Kubernetes secrets...$(NC)"
	@kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1 || true
	@if [ -f .secrets/clickhouse_password ]; then \
		kubectl create secret generic clickhouse-credentials \
			--from-file=password=.secrets/clickhouse_password \
			-n $(NAMESPACE) \
			--dry-run=client -o yaml | kubectl apply -f -; \
		echo "$(GREEN)✅ ClickHouse secret created$(NC)"; \
	fi
	@if [ -f .secrets/rabbitmq_password ]; then \
		kubectl create secret generic rabbitmq-credentials \
			--from-file=password=.secrets/rabbitmq_password \
			-n $(NAMESPACE) \
			--dry-run=client -o yaml | kubectl apply -f -; \
		echo "$(GREEN)✅ RabbitMQ secret created$(NC)"; \
	fi

secrets: create-secrets ## Alias for create-secrets

passwords: ## View current passwords
	@echo "$(YELLOW)🔑 Password Information$(NC)"
	@echo ""
	@if [ -f .secrets/clickhouse_password ]; then \
		echo "$(GREEN)ClickHouse Password (from file):$(NC)"; \
		cat .secrets/clickhouse_password; \
		echo ""; \
	fi
	@if [ -f .secrets/rabbitmq_password ]; then \
		echo "$(GREEN)RabbitMQ Password (from file):$(NC)"; \
		cat .secrets/rabbitmq_password; \
		echo ""; \
	fi
	@echo ""
	@if kubectl get secret clickhouse-credentials -n $(NAMESPACE) >/dev/null 2>&1; then \
		echo "$(GREEN)ClickHouse Secret (from K8s):$(NC)"; \
		kubectl get secret clickhouse-credentials -n $(NAMESPACE) -o jsonpath="{.data.password}" | base64 -d; \
		echo ""; \
	else \
		echo "$(YELLOW)ClickHouse secret not found in Kubernetes$(NC)"; \
	fi
	@echo ""
	@if kubectl get secret rabbitmq-credentials -n $(NAMESPACE) >/dev/null 2>&1; then \
		echo "$(GREEN)RabbitMQ Secret (from K8s):$(NC)"; \
		kubectl get secret rabbitmq-credentials -n $(NAMESPACE) -o jsonpath="{.data.password}" | base64 -d; \
		echo ""; \
	else \
		echo "$(YELLOW)RabbitMQ secret not found in Kubernetes$(NC)"; \
	fi

check-chart-sources: ## Test all available chart sources
	@echo "$(YELLOW)🔍 Testing chart source availability...$(NC)"
	@echo ""
	@echo "$(BLUE)Bitnami Helm Repository:$(NC)"
	@if helm repo add bitnami-test https://charts.bitnami.com/bitnami >/dev/null 2>&1 && helm repo update >/dev/null 2>&1; then \
		echo "$(GREEN)  ✅ Available$(NC)"; \
		helm repo remove bitnami-test >/dev/null 2>&1; \
	else \
		echo "$(RED)  ❌ Not available$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)OCI Registry:$(NC)"
	@if helm show chart oci://registry-1.docker.io/bitnamicharts/clickhouse --version 6.2.17 >/dev/null 2>&1; then \
		echo "$(GREEN)  ✅ Available$(NC)"; \
	else \
		echo "$(RED)  ❌ Not available$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Direct Downloads:$(NC)"
	@if curl -s -I "https://github.com/bitnami/charts/releases/download/clickhouse-6.2.17/clickhouse-6.2.17.tgz" | grep -q "200 OK"; then \
		echo "$(GREEN)  ✅ Available$(NC)"; \
	else \
		echo "$(RED)  ❌ Not available$(NC)"; \
	fi

deploy: create-secrets ## Deploy data stack (auto-detects best chart source)
	@echo "$(GREEN)🚀 Intelligent Data Stack Deployment$(NC)"
	@if [ ! -f k8s/data-stack/clickhouse/values.yaml ]; then \
		echo "$(RED)❌ Run 'make init' first to create config files$(NC)"; \
		exit 1; \
	fi
	@ENVIRONMENT=$(ENVIRONMENT) ./scripts/deploy.sh

quick-deploy: init create-secrets deploy ## Complete setup: init + secrets + deploy

destroy: ## Destroy data stack (WARNING: deletes data)
	@echo "$(RED)⚠️  WARNING: This will delete all data in $(NAMESPACE)!$(NC)"
	@echo "$(RED)This includes:$(NC)"
	@echo "  - All ClickHouse databases and tables"
	@echo "  - All RabbitMQ queues and messages"
	@echo "  - All persistent volume data"
	@echo "  - All secrets"
	@echo ""
	@read -p "Are you sure? Type 'DELETE' to confirm: " confirm; \
	if [ "$$confirm" = "DELETE" ]; then \
		echo "$(YELLOW)🗑️  Destroying data stack...$(NC)"; \
		helm uninstall $(CLICKHOUSE_RELEASE) -n $(NAMESPACE) 2>/dev/null || true; \
		helm uninstall $(RABBITMQ_RELEASE) -n $(NAMESPACE) 2>/dev/null || true; \
		kubectl delete pvc -l app.kubernetes.io/instance=$(CLICKHOUSE_RELEASE) -n $(NAMESPACE) 2>/dev/null || true; \
		kubectl delete pvc -l app.kubernetes.io/instance=$(RABBITMQ_RELEASE) -n $(NAMESPACE) 2>/dev/null || true; \
		kubectl delete secret clickhouse-credentials rabbitmq-credentials -n $(NAMESPACE) 2>/dev/null || true; \
		kubectl delete namespace $(NAMESPACE) --ignore-not-found; \
		rm -f .secrets/clickhouse_password .secrets/rabbitmq_password; \
		echo "$(GREEN)✅ Data stack destroyed$(NC)"; \
	else \
		echo "$(BLUE)❌ Destruction cancelled$(NC)"; \
	fi

status: ## Show detailed status of all services
	@echo "$(YELLOW)📊 Data Stack Status$(NC)"
	@echo ""
	@if kubectl get namespace $(NAMESPACE) >/dev/null 2>&1; then \
		echo "$(GREEN)Namespace: $(NAMESPACE) ✓$(NC)"; \
	else \
		echo "$(RED)Namespace: $(NAMESPACE) ✗$(NC)"; \
		exit 1; \
	fi
	@echo ""
	@echo "$(BLUE)Secrets:$(NC)"
	@kubectl get secrets -n $(NAMESPACE) 2>/dev/null | grep -E "(clickhouse|rabbitmq)" || echo "No secrets found"
	@echo ""
	@echo "$(BLUE)Pods:$(NC)"
	@kubectl get pods -n $(NAMESPACE) -o wide 2>/dev/null || echo "No pods found"
	@echo ""
	@echo "$(BLUE)Services:$(NC)"
	@kubectl get svc -n $(NAMESPACE) 2>/dev/null || echo "No services found"
	@echo ""
	@echo "$(BLUE)Persistent Volume Claims:$(NC)"
	@kubectl get pvc -n $(NAMESPACE) 2>/dev/null || echo "No PVCs found"
	@echo ""
	@echo "$(BLUE)Helm Releases:$(NC)"
	@helm list -n $(NAMESPACE) 2>/dev/null || echo "No Helm releases found"

health: ## Check health of services with detailed diagnostics
	@echo "$(YELLOW)🏥 Health Check$(NC)"
	@echo ""
	@echo "$(BLUE)ClickHouse Health:$(NC)"
	@if kubectl get pod -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}' >/dev/null 2>&1; then \
		CH_POD=$$(kubectl get pod -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
		if kubectl exec $$CH_POD -n $(NAMESPACE) -- clickhouse-client -q "SELECT 1" >/dev/null 2>&1; then \
			echo "$(GREEN)  ✅ ClickHouse is healthy$(NC)"; \
			kubectl exec $$CH_POD -n $(NAMESPACE) -- clickhouse-client -q "SELECT version(), uptime()"; \
		else \
			echo "$(RED)  ❌ ClickHouse is not responding$(NC)"; \
		fi \
	else \
		echo "$(RED)  ❌ ClickHouse pod not found$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)RabbitMQ Health:$(NC)"
	@if kubectl get pod -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}' >/dev/null 2>&1; then \
		RMQ_POD=$$(kubectl get pod -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
		if kubectl exec $$RMQ_POD -n $(NAMESPACE) -- rabbitmq-diagnostics check_running >/dev/null 2>&1; then \
			echo "$(GREEN)  ✅ RabbitMQ is healthy$(NC)"; \
			kubectl exec $$RMQ_POD -n $(NAMESPACE) -- rabbitmq-diagnostics status | head -10; \
		else \
			echo "$(RED)  ❌ RabbitMQ is not responding$(NC)"; \
		fi \
	else \
		echo "$(RED)  ❌ RabbitMQ pod not found$(NC)"; \
	fi

logs: ## Show logs from services
	@echo "$(YELLOW)📋 Service Logs$(NC)"
	@echo ""
	@echo "$(BLUE)ClickHouse Logs (last 20 lines):$(NC)"
	@kubectl logs -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) --tail=20 2>/dev/null || echo "No ClickHouse logs available"
	@echo ""
	@echo "$(BLUE)RabbitMQ Logs (last 20 lines):$(NC)"
	@kubectl logs -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) --tail=20 2>/dev/null || echo "No RabbitMQ logs available"

logs-follow: ## Follow logs from all services
	@echo "$(YELLOW)📋 Following Service Logs (Ctrl+C to stop)$(NC)"
	@kubectl logs -f -l app.kubernetes.io/name=clickhouse,app.kubernetes.io/name=rabbitmq -n $(NAMESPACE)

port-forward: ## Start port forwarding for local access
	@echo "$(YELLOW)🔗 Starting port forwarding...$(NC)"
	@./scripts/port-forward.sh

urls: ## Show connection URLs and credentials
	@echo "$(YELLOW)🔗 Connection Information$(NC)"
	@echo ""
	@if kubectl get svc $(CLICKHOUSE_RELEASE) -n $(NAMESPACE) >/dev/null 2>&1; then \
		echo "$(GREEN)ClickHouse:$(NC)"; \
		echo "  HTTP API: http://localhost:8123 (after port-forward)"; \
		echo "  Web UI: http://localhost:8123/play"; \
		echo "  TCP: localhost:9000"; \
		echo "  User: default"; \
		echo -n "  Password: "; \
		if [ -f .secrets/clickhouse_password ]; then \
			cat .secrets/clickhouse_password; \
		else \
			kubectl get secret clickhouse-credentials -n $(NAMESPACE) -o jsonpath="{.data.password}" 2>/dev/null | base64 -d || echo "Not available"; \
		fi; \
		echo ""; \
		echo "  In-cluster: $(CLICKHOUSE_RELEASE).$(NAMESPACE).svc.cluster.local:8123"; \
	else \
		echo "$(RED)ClickHouse not deployed$(NC)"; \
	fi
	@echo ""
	@if kubectl get svc $(RABBITMQ_RELEASE) -n $(NAMESPACE) >/dev/null 2>&1; then \
		echo "$(GREEN)RabbitMQ:$(NC)"; \
		echo "  Management UI: http://localhost:15672 (after port-forward)"; \
		echo "  AMQP: localhost:5672"; \
		echo "  User: admin"; \
		echo -n "  Password: "; \
		if [ -f .secrets/rabbitmq_password ]; then \
			cat .secrets/rabbitmq_password; \
		else \
			kubectl get secret rabbitmq-credentials -n $(NAMESPACE) -o jsonpath="{.data.password}" 2>/dev/null | base64 -d || echo "Not available"; \
		fi; \
		echo ""; \
		echo "  In-cluster: $(RABBITMQ_RELEASE).$(NAMESPACE).svc.cluster.local:5672"; \
	else \
		echo "$(RED)RabbitMQ not deployed$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)To start port forwarding: make port-forward$(NC)"

connect-clickhouse: ## Connect to ClickHouse CLI
	@echo "$(YELLOW)🗃️  Connecting to ClickHouse...$(NC)"
	@CH_POD=$$(kubectl get pod -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
	kubectl exec -it $$CH_POD -n $(NAMESPACE) -- clickhouse-client

connect-rabbitmq: ## Connect to RabbitMQ container
	@echo "$(YELLOW)🐰 Connecting to RabbitMQ...$(NC)"
	@RMQ_POD=$$(kubectl get pod -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
	kubectl exec -it $$RMQ_POD -n $(NAMESPACE) -- bash

restart: ## Restart all services
	@echo "$(YELLOW)🔄 Restarting services...$(NC)"
	@kubectl rollout restart deployment -n $(NAMESPACE)
	@echo "$(GREEN)✅ Restart initiated$(NC)"
	@echo "$(BLUE)Use 'make status' to check progress$(NC)"

clean: ## Clean up failed resources
	@echo "$(YELLOW)🧹 Cleaning up failed resources...$(NC)"
	@kubectl delete pods --field-selector=status.phase=Failed -n $(NAMESPACE) 2>/dev/null || true
	@helm list -n $(NAMESPACE) --failed -q | xargs -r helm delete -n $(NAMESPACE)
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

upgrade: ## Upgrade services with latest values
	@echo "$(YELLOW)⬆️  Upgrading services...$(NC)"
	@helm upgrade $(CLICKHOUSE_RELEASE) bitnami/clickhouse \
		-n $(NAMESPACE) \
		--values k8s/data-stack/clickhouse/values.yaml \
		--values helm/values/$(ENVIRONMENT)/clickhouse-values.yaml
	@helm upgrade $(RABBITMQ_RELEASE) bitnami/rabbitmq \
		-n $(NAMESPACE) \
		--values k8s/data-stack/rabbitmq/values.yaml \
		--values helm/values/$(ENVIRONMENT)/rabbitmq-values.yaml
	@echo "$(GREEN)✅ Upgrade completed$(NC)"

# Environment shortcuts
dev: ## Deploy development environment
	@$(MAKE) ENVIRONMENT=dev quick-deploy

staging: ## Deploy staging environment
	@$(MAKE) ENVIRONMENT=staging quick-deploy

prod: ## Deploy production environment
	@$(MAKE) ENVIRONMENT=production quick-deploy

# Quick actions
run: quick-deploy ## Alias for quick-deploy
start: port-forward ## Alias for port-forward
stop: ## Stop port forwarding
	@pkill -f "kubectl port-forward" || echo "No port forwards running"

# Development helpers
reset: ## Reset everything and redeploy (DEV ONLY)
	@if [ "$(ENVIRONMENT)" = "production" ]; then \
		echo "$(RED)❌ Reset not allowed in production!$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)🔄 Resetting development environment...$(NC)"
	@make destroy || true
	@sleep 5
	@make quick-deploy
	@echo "$(GREEN)✅ Reset completed$(NC)"
