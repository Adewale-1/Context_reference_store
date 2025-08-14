# Architecture Diagram Design Guidelines

## High-Level Design Principles for Technical Architecture Diagrams

### Visual Style

- **Background**: Light gray (`#F8F9FA`) instead of white for professional appearance
- **Text**: Always use bold fonts (`Arial Bold`) for better readability
- **Spacing**: Compact layout - `ranksep: "1.5"`, `nodesep: "0.8"` for clean presentation
- **Components**: Use generic `Server` components instead of technology-specific icons

### Clinical Workflow Style Layout

- **Numbered Steps**: Clear sequence (1, 2, 3, 4) with descriptive headers
- **Clusters**: Rounded corners with colored backgrounds for visual grouping
- **Flow Direction**: Left-to-right (`LR`) for natural reading pattern

### Color Scheme (Material Design Inspired)

- **Step 1**: Blue theme - `#E3F2FD` background, `#1976D2` border
- **Step 2**: Purple theme - `#F3E5F5` background, `#7B1FA2` border
- **Step 3**: Orange theme - `#FFF3E0` background, `#F57C00` border
- **Step 4**: Green theme - `#E8F5E8` background, `#2E7D32` border

### Component Design

- **Detailed Descriptions**: Use bullet points (•) for feature lists
- **Performance Metrics**: Prominently display quantified benefits
- **Component Fill Colors**: Light tints matching cluster themes
- **No Technology Icons**: Avoid Redis, PostgreSQL, AWS icons - use generic shapes

### Arrow Styling

- **Bold Workflow**: `penwidth: "3"` for main data flow
- **Colored Labels**: Descriptive text with semantic colors
- **Varied Styles**:
  - Solid bold: Primary workflow
  - Dashed: Response/feedback loops
  - Dotted: Oversight/management connections

### Content Guidelines

- **Technical Details**: Include specific metrics and capabilities
- **Clear Labeling**: Each arrow should have descriptive labels
- **Performance Focus**: Highlight speed, memory, and efficiency gains
- **Professional Tone**: Technical but accessible language

### Layout Rules

- **Overseeing Components**: Float management/cache components like clinical oversight
- **Benefits Box**: Separate cluster for performance metrics
- **Clean Connections**: Avoid crossing arrows where possible
- **Consistent Spacing**: Uniform margins and padding throughout

This design creates professional, technical diagrams suitable for documentation, presentations, and architectural reviews.
