.PHONY: help init deploy destroy status logs clean

# Configuration
NAMESPACE := data-stack
ENVIRONMENT ?= dev

help: ## Show this help
	@echo "ClickHouse Data Stack Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: ## Initialize configuration files from templates
	@echo "🔧 Initializing configuration files..."
	@if [ ! -f k8s/data-stack/clickhouse/values.yaml ]; then \
		cp k8s/data-stack/clickhouse/values-template.yaml k8s/data-stack/clickhouse/values.yaml; \
		echo "✅ Created ClickHouse values.yaml - please edit credentials"; \
	fi
	@if [ ! -f k8s/data-stack/rabbitmq/values.yaml ]; then \
		cp k8s/data-stack/rabbitmq/values-template.yaml k8s/data-stack/rabbitmq/values.yaml; \
		echo "✅ Created RabbitMQ values.yaml - please edit credentials"; \
	fi
	@if [ ! -f scripts/deploy.sh ]; then \
		cp scripts/deploy-template.sh scripts/deploy.sh; \
		chmod +x scripts/deploy.sh; \
		echo "✅ Created deploy.sh"; \
	fi
	@if [ ! -f scripts/port-forward.sh ]; then \
		cp scripts/port-forward-template.sh scripts/port-forward.sh; \
		chmod +x scripts/port-forward.sh; \
		echo "✅ Created port-forward.sh"; \
	fi

deploy: ## Deploy data stack
	@ENVIRONMENT=$(ENVIRONMENT) ./scripts/deploy.sh

destroy: ## Destroy data stack (WARNING: deletes data)
	@echo "⚠️  WARNING: This will delete all data in $(NAMESPACE)!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@helm uninstall clickhouse -n $(NAMESPACE) 2>/dev/null || true
	@helm uninstall rabbitmq -n $(NAMESPACE) 2>/dev/null || true
	@kubectl delete namespace $(NAMESPACE) --ignore-not-found

status: ## Show status of all services
	@echo "📊 Data Stack Status:"
	@kubectl get pods -n $(NAMESPACE) 2>/dev/null || echo "Namespace $(NAMESPACE) not found"
	@echo "\n🔗 Services:"
	@kubectl get svc -n $(NAMESPACE) 2>/dev/null || true
	@echo "\n💾 Storage:"
	@kubectl get pvc -n $(NAMESPACE) 2>/dev/null || true

logs: ## Show logs from services
	@echo "📋 Recent logs:"
	@kubectl logs -l app.kubernetes.io/name=clickhouse -n $(NAMESPACE) --tail=20 2>/dev/null || true
	@kubectl logs -l app.kubernetes.io/name=rabbitmq -n $(NAMESPACE) --tail=20 2>/dev/null || true

port-forward: ## Start port forwarding
	@./scripts/port-forward.sh

clean: ## Clean up failed resources
	@kubectl delete pods --field-selector=status.phase=Failed -n $(NAMESPACE) 2>/dev/null || true
	@helm list -n $(NAMESPACE) --failed -q | xargs -r helm delete -n $(NAMESPACE)

dev: ## Quick deploy for development
	@$(MAKE) ENVIRONMENT=dev deploy

prod: ## Deploy for production
	@$(MAKE) ENVIRONMENT=production deploy
