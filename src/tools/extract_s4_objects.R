#!/usr/bin/env Rscript
# R Script to Extract S4 Objects from PAMpal exStudy.rda
# 
# This script loads complex S4 objects from R data files and extracts their
# components into simpler R data structures that can be easily converted to Python.
#
# Usage: Rscript extract_s4_objects.R <input_file> <output_dir>

# Load required libraries
if (!require("methods", quietly = TRUE)) {
    stop("methods package not available")
}

# Suppress package loading messages during load()
suppressPackageStartupMessages({
    # We'll work with the raw S4 structure without requiring PAMpal package
})

# Function to safely extract slot from S4 object
extract_slot_safe <- function(obj, slot_name) {
    tryCatch({
        # Try different methods to access slots
        if (methods::hasSlot(obj, slot_name)) {
            return(methods::slot(obj, slot_name))
        } else if (slot_name %in% slotNames(obj)) {
            return(slot(obj, slot_name))
        } else {
            # Try direct access via @ operator
            return(eval(parse(text = paste0("obj@", slot_name))))
        }
    }, error = function(e) {
        cat("Warning: Could not extract slot", slot_name, ":", e$message, "\n")
        return(NULL)
    })
}

# Function to convert S4 object to list recursively
s4_to_list <- function(obj) {
    if (isS4(obj)) {
        # Get all slot names - try multiple methods
        slot_names <- tryCatch({
            methods::slotNames(obj)
        }, error = function(e) {
            # If slotNames fails, try alternative approach
            tryCatch({
                names(attributes(obj))
            }, error = function(e2) {
                cat("Warning: Could not get slot names for S4 object\n")
                character(0)
            })
        })
        
        result <- list()
        
        cat("Converting S4 object with slots:", paste(slot_names, collapse = ", "), "\n")
        
        for (slot_name in slot_names) {
            slot_value <- extract_slot_safe(obj, slot_name)
            if (!is.null(slot_value)) {
                # Recursively convert nested S4 objects
                result[[slot_name]] <- s4_to_list(slot_value)
            }
        }
        
        # Add class information
        result[[".__class__"]] <- class(obj)
        return(result)
        
    } else if (is.list(obj)) {
        # Convert list elements recursively
        return(lapply(obj, s4_to_list))
        
    } else {
        # Return primitive types as-is
        return(obj)
    }
}

# Function to extract AcousticStudy components
extract_acoustic_study <- function(study_obj) {
    cat("Extracting AcousticStudy object...\n")
    
    # Initialize result structure
    result <- list(
        metadata = list(
            extraction_timestamp = Sys.time(),
            original_class = class(study_obj),
            r_version = R.version.string
        )
    )
    
    # Extract main study properties
    if (isS4(study_obj)) {
        study_slots <- methods::slotNames(study_obj)
        cat("Study slots found:", paste(study_slots, collapse = ", "), "\n")
        
        # Extract each slot
        for (slot_name in study_slots) {
            cat("Extracting slot:", slot_name, "\n")
            slot_value <- extract_slot_safe(study_obj, slot_name)
            
            if (!is.null(slot_value)) {
                # Handle specific slot types
                if (slot_name == "events") {
                    result$events <- extract_events(slot_value)
                } else if (slot_name == "pps") {
                    result$pps <- extract_pps(slot_value)
                } else {
                    result[[slot_name]] <- s4_to_list(slot_value)
                }
            }
        }
    } else {
        cat("Warning: Object is not S4, treating as regular list\n")
        result$raw_data <- s4_to_list(study_obj)
    }
    
    return(result)
}

# Function to extract Events (AcousticEvent objects)
extract_events <- function(events_obj) {
    cat("Extracting events...\n")
    
    if (is.list(events_obj)) {
        result <- list()
        
        for (event_name in names(events_obj)) {
            cat("Extracting event:", event_name, "\n")
            event <- events_obj[[event_name]]
            
            if (isS4(event)) {
                event_slots <- methods::slotNames(event)
                cat("  Event slots:", paste(event_slots, collapse = ", "), "\n")
                
                event_data <- list()
                for (slot_name in event_slots) {
                    slot_value <- extract_slot_safe(event, slot_name)
                    
                    if (slot_name == "detectors") {
                        event_data$detectors <- extract_detectors(slot_value)
                    } else {
                        event_data[[slot_name]] <- s4_to_list(slot_value)
                    }
                }
                result[[event_name]] <- event_data
            } else {
                result[[event_name]] <- s4_to_list(event)
            }
        }
        
        return(result)
    } else {
        return(s4_to_list(events_obj))
    }
}

# Function to extract detector data
extract_detectors <- function(detectors_obj) {
    cat("Extracting detectors...\n")
    
    if (is.list(detectors_obj)) {
        result <- list()
        
        for (detector_name in names(detectors_obj)) {
            cat("  Extracting detector:", detector_name, "\n")
            detector_data <- detectors_obj[[detector_name]]
            
            # Convert to data.frame if possible
            if (is.data.frame(detector_data)) {
                result[[detector_name]] <- detector_data
            } else {
                result[[detector_name]] <- s4_to_list(detector_data)
            }
        }
        
        return(result)
    } else {
        return(s4_to_list(detectors_obj))
    }
}

# Function to extract PPS (PAMpalSettings)
extract_pps <- function(pps_obj) {
    cat("Extracting PAMpalSettings...\n")
    
    if (isS4(pps_obj)) {
        return(s4_to_list(pps_obj))
    } else {
        return(pps_obj)
    }
}

# Function to save extracted data in multiple formats
save_extracted_data <- function(extracted_data, output_dir, base_name) {
    # Create output directory if it doesn't exist
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }
    
    # Save as RData file (simplified structure)
    rdata_file <- file.path(output_dir, paste0(base_name, "_extracted.RData"))
    save(extracted_data, file = rdata_file)
    cat("Saved extracted data to:", rdata_file, "\n")
    
    # Save individual components as CSV when possible
    if ("events" %in% names(extracted_data)) {
        events_dir <- file.path(output_dir, "events")
        if (!dir.exists(events_dir)) {
            dir.create(events_dir)
        }
        
        for (event_name in names(extracted_data$events)) {
            event_data <- extracted_data$events[[event_name]]
            
            if ("detectors" %in% names(event_data)) {
                for (detector_name in names(event_data$detectors)) {
                    detector_data <- event_data$detectors[[detector_name]]
                    
                    if (is.data.frame(detector_data)) {
                        csv_file <- file.path(events_dir, paste0(event_name, "_", detector_name, ".csv"))
                        write.csv(detector_data, csv_file, row.names = FALSE)
                        cat("Saved detector data to:", csv_file, "\n")
                    }
                }
            }
        }
    }
    
    # Save metadata
    metadata_file <- file.path(output_dir, paste0(base_name, "_metadata.txt"))
    writeLines(c(
        paste("Extraction timestamp:", Sys.time()),
        paste("R version:", R.version.string),
        paste("Original file:", base_name),
        paste("Number of top-level components:", length(extracted_data)),
        paste("Component names:", paste(names(extracted_data), collapse = ", "))
    ), metadata_file)
    
    return(invisible(extracted_data))
}

# Main function
main <- function() {
    # Parse command line arguments
    args <- commandArgs(trailingOnly = TRUE)
    
    if (length(args) < 2) {
        cat("Usage: Rscript extract_s4_objects.R <input_file> <output_dir>\n")
        cat("Example: Rscript extract_s4_objects.R exStudy.rda ./extracted_data/\n")
        quit(status = 1)
    }
    
    input_file <- args[1]
    output_dir <- args[2]
    
    cat("PAMpal S4 Object Extractor\n")
    cat("==========================\n")
    cat("Input file:", input_file, "\n")
    cat("Output directory:", output_dir, "\n\n")
    
    # Check if input file exists
    if (!file.exists(input_file)) {
        cat("Error: Input file does not exist:", input_file, "\n")
        quit(status = 1)
    }
    
    # Load the R data file
    cat("Loading R data file...\n")
    tryCatch({
        # Suppress package loading errors and warnings during load
        old_options <- options(warn = -1)
        on.exit(options(old_options))
        
        # Load the .rda file
        loaded_objects <- load(input_file)
        cat("Loaded objects:", paste(loaded_objects, collapse = ", "), "\n")
        
        # Process each loaded object
        for (obj_name in loaded_objects) {
            cat("\nProcessing object:", obj_name, "\n")
            obj <- get(obj_name)
            
            cat("Object class:", class(obj), "\n")
            cat("Object type:", typeof(obj), "\n")
            
            # Extract the object
            if (isS4(obj)) {
                cat("Extracting S4 object...\n")
                extracted_data <- extract_acoustic_study(obj)
            } else {
                cat("Converting non-S4 object...\n")
                extracted_data <- s4_to_list(obj)
            }
            
            # Save the extracted data
            base_name <- tools::file_path_sans_ext(basename(input_file))
            save_extracted_data(extracted_data, output_dir, paste0(base_name, "_", obj_name))
        }
        
        cat("\nExtraction completed successfully!\n")
        
    }, error = function(e) {
        cat("Error loading or processing file:", e$message, "\n")
        quit(status = 1)
    })
}

# Run main function if script is executed directly
if (!interactive()) {
    main()
}