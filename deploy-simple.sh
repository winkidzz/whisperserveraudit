#!/bin/bash

# Simple Deployment Script using CapRover CLI
# Deploys the app using the registry image

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY_URL="ishworksregistry.captain.ishworks.website"
APP_NAME="whispercaprover"
TAG="latest"
CAPROVER_URL="https://captain.captain.ishworks.website"

echo -e "${BLUE}🚀 Simple WhisperCapRover Deployment${NC}"
echo "========================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if CapRover CLI is installed
    if ! command -v caprover &> /dev/null; then
        print_error "CapRover CLI not found. Please install it first: npm install -g caprover"
        exit 1
    fi
    
    print_status "Prerequisites check passed!"
}

# Deploy using CapRover CLI
deploy_with_cli() {
    print_status "Deploying using CapRover CLI..."
    
    # Deploy using the registry image
    caprover deploy \
        --caproverUrl "${CAPROVER_URL}" \
        --caproverPassword "prasanna" \
        --caproverApp "${APP_NAME}" \
        --imageName "${REGISTRY_URL}/${APP_NAME}:${TAG}"
    
    print_status "Deployment completed!"
}

# Show deployment info
show_deployment_info() {
    print_status "Deployment completed!"
    print_status "Image: ${REGISTRY_URL}/${APP_NAME}:${TAG}"
    print_status "App URL: ${CAPROVER_URL}/${APP_NAME}"
    print_status "Health check: ${CAPROVER_URL}/${APP_NAME}/health"
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    - Deploy using CapRover CLI (default)"
    echo "  help      - Show this help"
    echo ""
    echo "This script deploys the WhisperCapRover app using CapRover CLI."
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            deploy_with_cli
            show_deployment_info
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 