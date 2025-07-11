#!/usr/bin/env Rscript
# Raw S4 Object Extractor - No Package Dependencies
# 
# This script extracts S4 objects without loading their class packages,
# treating them as raw data structures.

# Intercept package loading
.original_library <- library
library <- function(...) {
    cat("Intercepted library call, skipping...\n")
    # Don't actually load anything
}

# Intercept require calls
.original_require <- require  
require <- function(...) {
    cat("Intercepted require call, skipping...\n")
    return(FALSE)
}

# Function to extract raw attributes from S4 object
extract_raw_s4 <- function(obj) {
    if (isS4(obj)) {
        cat("Extracting raw S4 object attributes...\n")
        
        # Get all attributes
        attrs <- attributes(obj)
        result <- list()
        
        # Extract the class info
        result[[".__class__"]] <- class(obj)
        
        # Try to extract data without using slots
        for (attr_name in names(attrs)) {
            cat("  Extracting attribute:", attr_name, "\n")
            
            if (attr_name == "class") {
                result[[attr_name]] <- attrs[[attr_name]]
            } else {
                # Try to extract the attribute value
                tryCatch({
                    attr_value <- attrs[[attr_name]]
                    result[[attr_name]] <- extract_raw_s4(attr_value)
                }, error = function(e) {
                    cat("    Warning: Could not extract", attr_name, ":", e$message, "\n")
                    result[[attr_name]] <- paste("Error extracting:", e$message)
                })
            }
        }
        
        # Try to access the object's internal structure directly
        tryCatch({
            # Get the object's internal representation
            obj_str <- capture.output(str(obj))
            result[[".__structure__"]] <- obj_str
        }, error = function(e) {
            cat("Could not capture structure\n")
        })
        
        return(result)
        
    } else if (is.list(obj)) {
        return(lapply(obj, extract_raw_s4))
    } else {
        return(obj)
    }
}

# Main extraction function
main <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    
    if (length(args) < 2) {
        cat("Usage: Rscript extract_s4_raw.R <input_file> <output_dir>\n")
        quit(status = 1)
    }
    
    input_file <- args[1]
    output_dir <- args[2]
    
    cat("Raw S4 Object Extractor\n")
    cat("=======================\n")
    cat("Input file:", input_file, "\n")
    cat("Output directory:", output_dir, "\n\n")
    
    # Create output directory
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }
    
    # Override package loading completely
    options(warn = -1)
    
    tryCatch({
        # Load without triggering package loads
        cat("Loading file with package loading disabled...\n")
        
        # Create isolated environment
        load_env <- new.env()
        
        # Load into isolated environment
        loaded_objects <- load(input_file, envir = load_env)
        cat("Loaded objects:", paste(loaded_objects, collapse = ", "), "\n")
        
        # Process each object
        for (obj_name in loaded_objects) {
            cat("\nProcessing object:", obj_name, "\n")
            obj <- get(obj_name, envir = load_env)
            
            cat("Object class:", class(obj), "\n")
            cat("Is S4:", isS4(obj), "\n")
            
            # Extract raw data
            extracted_data <- extract_raw_s4(obj)
            
            # Add metadata
            result <- list(
                object_name = obj_name,
                extraction_timestamp = Sys.time(),
                r_version = R.version.string,
                original_class = class(obj),
                is_s4 = isS4(obj),
                extracted_data = extracted_data
            )
            
            # Save as RData
            output_file <- file.path(output_dir, paste0(obj_name, "_raw_extracted.RData"))
            save(result, file = output_file)
            cat("Saved to:", output_file, "\n")
            
            # Try to save individual components as CSV
            if (is.list(extracted_data)) {
                for (component_name in names(extracted_data)) {
                    component_data <- extracted_data[[component_name]]
                    
                    if (is.data.frame(component_data)) {
                        csv_file <- file.path(output_dir, paste0(obj_name, "_", component_name, ".csv"))
                        write.csv(component_data, csv_file, row.names = FALSE)
                        cat("Saved component to:", csv_file, "\n")
                    }
                }
            }
        }
        
        cat("\nRaw extraction completed successfully!\n")
        
    }, error = function(e) {
        cat("Error during extraction:", e$message, "\n")
        quit(status = 1)
    })
}

# Run if executed directly
if (!interactive()) {
    main()
}