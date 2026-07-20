terraform {
  required_providers {
    oci = { source = "oracle/oci", version = "~> 6.0" }
  }
}

variable "compartment_id" {
  type        = string
  description = "OCI compartment OCID for all resources"
}

variable "tenancy_ocid" {
  type        = string
  description = "OCI tenancy OCID (used to look up availability domains)"
}

variable "ubuntu_image_id" {
  type        = string
  description = "OCID of the Ubuntu 22.04 ARM64 image (resolved by root module)"
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key content for instance access"
}

variable "instance_name" {
  type        = string
  description = "Display name for the compute instance"
}

variable "instance_ocpus" {
  type        = number
  description = "Number of OCPUs for the A1.Flex instance"
}

variable "instance_memory_gb" {
  type        = number
  description = "RAM in GB for the A1.Flex instance"
}

variable "boot_volume_size_gb" {
  type        = number
  description = "Boot volume size in GB"
}
