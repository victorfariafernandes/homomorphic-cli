variable "tenancy_ocid" {
  type        = string
  description = "OCI tenancy OCID"
}

variable "user_ocid" {
  type        = string
  description = "IAM user OCID"
}

variable "fingerprint" {
  type        = string
  description = "API key fingerprint"
}

variable "private_key_path" {
  type        = string
  default     = "~/.oci/oci_api_key.pem"
  description = "Path to the OCI API private key file"
}

variable "region" {
  type        = string
  default     = "us-ashburn-1"
  description = "OCI region"
}

variable "compartment_id" {
  type        = string
  description = "OCI compartment OCID for all resources"
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key content for instance access"
}

variable "ssh_private_key_path" {
  type        = string
  description = "Path to SSH private key matching ssh_public_key — used by Ansible to connect to the instance"
}

variable "instance_ocpus" {
  type        = number
  default     = 4
  description = "Number of OCPUs for the A1.Flex instance (free tier max: 4; parallel k=40 experiment: 64 — see parallel.tfvars.example)"
}

variable "instance_memory_gb" {
  type        = number
  default     = 24
  description = "RAM in GB for the A1.Flex instance (free tier max: 24; parallel k=40 experiment: 128 — see parallel.tfvars.example)"
}

variable "boot_volume_size_gb" {
  type        = number
  default     = 100
  description = "Boot volume size in GB"
}

variable "instance_name" {
  type        = string
  default     = "fhe-lssvm"
  description = "Display name for the compute instance"
}

variable "openfhe_version" {
  type        = string
  default     = "v1.5.1"
  description = "Git tag of openfhe-development to build on the instance"
}
