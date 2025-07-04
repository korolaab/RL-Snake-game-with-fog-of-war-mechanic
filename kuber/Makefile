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
	@echo "ClickHouse Data Stack - Available Commands"
	@echo ""
	@echo "Setup & Deployment:"
	@grep -E '^(init|secrets|deploy|quick-deploy):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Management & Monitoring:"
	@grep -E '^(status|health|logs|how-to-connect):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Utilities:"
	@grep -E '^(passwords|create-tables|destroy|clean):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Getting Started (Cold Setup):$(NC)"
	@echo "  1. make init           Create namespace and config files"
	@echo "  2. make secrets        Generate passwords and create K8s secrets"
	@echo "  3. make deploy         Deploy ClickHouse and RabbitMQ"
	@echo "  4. make create-tables  Initialize database tables"
	@echo "  5. make status         Verify everything is running"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make quick-deploy      Run all setup steps automatically"
	@echo ""
	@echo "$(YELLOW)Daily Operations:$(NC)"
	@echo "  make status           Check deployment status"
	@echo "  make health           Run health checks"
	@echo "  make how-to-connect   Get connection details"

init: ## Initialize namespace
	@echo "$(YELLOW)🔧 Creating namespace $(NAMESPACE)...$(NC)"
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

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

deploy: create-secrets ## Deploy data stack (auto-detects best chart source)
	@echo "$(GREEN) Data Stack Deployment$(NC)"
	@if [ ! -f k8s/data-stack/clickhouse/values.yaml ]; then \
		echo "$(RED)❌ Run 'make init' first to create config files$(NC)"; \
		exit 1; \
	fi
	@helm install rabbitmq oci://registry-1.docker.io/bitnamicharts/rabbitmq  -n $(NAMESPACE)   --values k8s/data-stack/rabbitmq/values.yaml -f k8s/data-stack/rabbitmq/values-secrets.yaml --wait
	@helm install clickhouse oci://registry-1.docker.io/bitnamicharts/clickhouse   -n $(NAMESPACE)   --values k8s/data-stack/clickhouse/values-minimal.yaml --wait

create-tables:
	@echo "$(GREEN) Creating clickhouse tables...$(NC)"
	@echo ""
	@if kubectl get pod -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}' >/dev/null 2>&1; then \
		CH_POD=$$(kubectl get pod -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
		CLICKHOUSE_PASSWORD=$$(cat .secrets/clickhouse_password); \
		if kubectl exec $$CH_POD -n $(NAMESPACE) -- clickhouse-client --password $$CLICKHOUSE_PASSWORD -q "SELECT 1" >/dev/null 2>&1; then \
			echo "$(GREEN)  ✅ ClickHouse is healthy$(NC)"; \
			kubectl exec $$CH_POD -n $(NAMESPACE) -- clickhouse-client --password $$CLICKHOUSE_PASSWORD --query "$$(cat scripts/tables.sql)"; \
			echo "$(GREEN)  ✅ Tables Created$(NC)"; \
		else \
			echo "$(RED)  ❌ ClickHouse is not responding$(NC)"; \
		fi \
	else \
		echo "$(RED)  ❌ ClickHouse pod not found$(NC)"; \
	fi

setup-tech-creds:
	@echo "$(GREEN) Setuping tech credentials... $(NC)"
	@echo ""
	@if kubectl get pod -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}' >/dev/null 2>&1; then \
		RMQ_POD=$$(kubectl get pod -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}'); \
		if kubectl exec $$RMQ_POD -n $(NAMESPACE) -- rabbitmq-diagnostics check_running >/dev/null 2>&1; then \
			echo "$(GREEN)  ✅ RabbitMQ is healthy$(NC)"; \
			kubectl exec $$RMQ_POD -n $(NAMESPACE) -- rabbitmq-diagnostics status | head -10; \
    		kubectl -n $(NAMESPACE) exec -it $$RMQ_POD -- rabbitmqctl add_user tech tech; \
			kubectl -n $(NAMESPACE) exec -it $$RMQ_POD -- rabbitmqctl set_permissions -p / tech ".*" ".*" ".*"; \
			kubectl -n $(NAMESPACE) exec -it $$RMQ_POD -- rabbitmqctl set_user_tags tech ""; \
			echo "$(GREEN)  tech credentials setuped$(NC)"; \
		else \
			echo "$(RED)  ❌ RabbitMQ is not responding$(NC)"; \
		fi \
	else \
		echo "$(RED)  ❌ RabbitMQ pod not found$(NC)"; \
	fi
	
quick-deploy: init create-secrets deploy health setup-tech-creds create-tables ## Complete setup: init + secrets + deploy

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
		CLICKHOUSE_PASSWORD=$$(cat .secrets/clickhouse_password); \
		if kubectl exec $$CH_POD -n $(NAMESPACE) -- clickhouse-client --password $$CLICKHOUSE_PASSWORD -q "SELECT 1" >/dev/null 2>&1; then \
			echo "$(GREEN)  ✅ ClickHouse is healthy$(NC)"; \
			kubectl exec $$CH_POD -n $(NAMESPACE) -- clickhouse-client --password $$CLICKHOUSE_PASSWORD -q "SELECT version(), uptime()"; \
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

how-to-connect: ## Show connection URLs and credentials
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
		helm status -n $(NAMESPACE) $(RABBITMQ_RELEASE); \
		echo ""; \
		echo "  In-cluster: $(RABBITMQ_RELEASE).$(NAMESPACE).svc.cluster.local:5672"; \
	else \
		echo "$(RED)RabbitMQ not deployed$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)To start port forwarding: make port-forward$(NC)"

